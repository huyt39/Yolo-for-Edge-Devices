import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from select_features import extract_features
from sklearn.metrics import accuracy_score
import wandb

class GenderDataset(Dataset):
    def __init__(self, rois_path, annotation_path, transform):
        self.samples = []
        self.transform = transform
        
        for folder in os.listdir(rois_path):
            folder_path = os.path.join(rois_path, folder)
            annotation_file = os.path.join(annotation_path, folder + ".txt")
            
            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
            selected_features = [1, 4]  # ID and Gender
            labels_dict = extract_features([annotation_file], selected_features)
            
            for id in os.listdir(folder_path):
                if id == "-1":
                    continue
                subfolder = os.path.join(folder_path, id)
                if os.path.isdir(subfolder):
                    gender = labels_dict.get(id)
                    if gender is None or gender not in [0, 1]:
                        continue
                    for file in os.listdir(subfolder):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(subfolder, file)
                            self.samples.append((img_path, gender))
            print(f"Done: {folder}")
    
    def __getitem__(self, index):
        path, gender = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, gender
    
    def __len__(self):
        return len(self.samples)

def GenderClassification(rois_path, annotation_path, transform, batch_size, fraction=0.01, use_pretrained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_dataset = GenderDataset(rois_path, annotation_path, transform)
    epochs = 30
    patience = 3  # Early stopping
    no_improvement_epochs = 0
    
    model = models.mobilenet_v2(pretrained=use_pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, 2)  # Binary classification
    model.to(device)
    
    wandb.finish()
    wandb.init(project="gender-classification", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.0001,
        "architecture": "MobileNetV2"
    })
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_valid_accuracy = 0.0
    best_model_wts = None
    
    for epoch in range(epochs):
        total_indices = torch.randperm(len(image_dataset))
        reduced_size = int(len(image_dataset) * fraction)
        selected_indices = total_indices[:reduced_size]
        subset_dataset = torch.utils.data.Subset(image_dataset, selected_indices.tolist())
        
        train_size = int(0.7 * reduced_size)
        valid_size = int(0.15 * reduced_size)
        test_size = reduced_size - train_size - valid_size
        
        train_set, valid_set, test_set = random_split(subset_dataset, [train_size, valid_size, test_size])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        train_accuracy = correct_predictions / total_predictions
        train_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy})
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        model.eval()
        valid_loss = 0.0
        valid_correct_predictions = 0
        valid_total_predictions = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_correct_predictions += (predicted == labels).sum().item()
                valid_total_predictions += labels.size(0)
        
        valid_accuracy = valid_correct_predictions / valid_total_predictions if valid_total_predictions > 0 else 0
        valid_loss /= len(valid_loader)
        wandb.log({"epoch": epoch + 1, "valid_loss": valid_loss, "valid_accuracy": valid_accuracy})
        print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}")
        
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), "best_model_mobilenet.pth")
            print(f"Best model saved with validation accuracy: {best_valid_accuracy:.4f}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break
    
    wandb.finish()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rois_path = "/mnt/e/workspace/Dataset/P-DESTR/rois/jpg_Extracted_PIDS"
    annotation_path = "/mnt/e/workspace/Dataset/P-DESTR/dataset/P-DESTRE/annotation"
    batch_size = 32
    
    GenderClassification(rois_path, annotation_path, transform, batch_size, use_pretrained=False)