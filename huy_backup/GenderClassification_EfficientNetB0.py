import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score
import wandb


def extract_gender_labels(annotation_path: str, id_col: int = 1, gender_col: int = 10) -> dict:
    """
    Parse P-DESTRE annotation file and extract gender per track ID.

    P-DESTRE annotation columns:
        0-6:   frame_id, track_id, x, y, w, h, confidence
        7-9:   yaw, pitch, roll
        10-25: soft-biometric attributes (gender first)

    Returns:
        dict mapping str(track_id) -> gender (0 or 1)
    """
    labels = {}
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(id_col, gender_col):
                continue
            try:
                tid = parts[id_col].strip()
                gender = int(parts[gender_col].strip())
            except (ValueError, IndexError):
                continue
            if gender in (0, 1) and tid not in labels:
                labels[tid] = gender
    return labels


class GenderDataset(Dataset):
    def __init__(self, rois_path, annotation_path, transform):
        self.samples = []
        self.transform = transform

        for folder in os.listdir(rois_path):
            folder_path = os.path.join(rois_path, folder)
            if not os.path.isdir(folder_path):
                continue
            annotation_file = os.path.join(annotation_path, folder + ".txt")

            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

            labels_dict = extract_gender_labels(annotation_file)

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


def GenderClassification(rois_path, annotation_path, train_transform, val_transform,
                         batch_size, train_ratio=0.7, val_ratio=0.15, use_pretrained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    patience = 5

    # --- Load full dataset with val transform (just for splitting) ---
    full_dataset = GenderDataset(rois_path, annotation_path, val_transform)
    print(f"Total samples: {len(full_dataset)}")

    # --- Split once ---
    n = len(full_dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=generator)

    # Apply train augmentation by wrapping the subset
    train_set.dataset = GenderDataset.__new__(GenderDataset)
    train_set.dataset.samples = full_dataset.samples
    train_set.dataset.transform = train_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Split: train={train_size}, val={val_size}, test={test_size}")

    # --- Model ---
    weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    # --- Training setup ---
    wandb.finish()
    wandb.init(project="gender-classification", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.0001,
        "architecture": "EfficientNet-B0",
        "pretrained": use_pretrained,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    })

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_accuracy = 0.0
    no_improvement_epochs = 0

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_loss /= len(val_loader)

        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
            "learning_rate": lr,
        })
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} | LR: {lr:.6f}")

        scheduler.step()

        # --- Early stopping ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model_efficientnet_b0.pth")
            print(f"  -> Best model saved (val acc: {best_val_accuracy:.4f})")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # --- Test evaluation ---
    model.load_state_dict(torch.load("best_model_efficientnet_b0.pth", weights_only=True))
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    wandb.log({"test_accuracy": test_accuracy})
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    wandb.finish()


if __name__ == '__main__':
    # EfficientNet-B0 expects 224x224 input
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rois_path = "/mnt/e/workspace/Dataset/P-DESTR/rois/jpg_Extracted_PIDS"
    annotation_path = "/mnt/e/workspace/Dataset/P-DESTR/dataset/P-DESTRE/annotation"
    batch_size = 32

    GenderClassification(rois_path, annotation_path, train_transform, val_transform,
                         batch_size, use_pretrained=True)
