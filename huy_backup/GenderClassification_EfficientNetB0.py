import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report
from scipy.io import loadmat
import wandb


# PA-100K attribute order (26 binary attributes; "Female" at index 0).
PA100K_ATTRIBUTES = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18",
    "Front", "Side", "Back",
    "Hat", "Glasses",
    "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern",
    "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
]


def load_pa100k_split(annotation_mat_path: str, split: str, attribute: str = "Female") -> list:
    """Return list of (image_filename, label) for PA-100K `split` ('train'/'val'/'test')."""
    mat = loadmat(annotation_mat_path)
    names = mat[f"{split}_images_name"].squeeze()
    labels = mat[f"{split}_label"]
    attr_idx = PA100K_ATTRIBUTES.index(attribute)
    return [(str(name[0]), int(labels[i, attr_idx])) for i, name in enumerate(names)]


class PA100KGenderDataset(Dataset):
    def __init__(self, images_dir, annotation_mat_path, split, transform):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = load_pa100k_split(annotation_mat_path, split, attribute="Female")

    def __getitem__(self, index):
        name, label = self.samples[index]
        image = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)


def GenderClassification(images_dir, annotation_mat_path, train_transform, val_transform,
                         batch_size, use_pretrained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    patience = 5

    # --- Datasets (official PA-100K train/val/test splits) ---
    train_set = PA100KGenderDataset(images_dir, annotation_mat_path, "train", train_transform)
    val_set = PA100KGenderDataset(images_dir, annotation_mat_path, "val", val_transform)
    test_set = PA100KGenderDataset(images_dir, annotation_mat_path, "test", val_transform)
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Model ---
    weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    # --- Training setup ---
    wandb.finish()
    wandb.init(project="pa100k-gender-classification", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.0001,
        "architecture": "EfficientNet-B0",
        "pretrained": use_pretrained,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
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
    test_preds, test_true = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().tolist())
            test_true.extend(labels.cpu().tolist())

    test_accuracy = sum(p == t for p, t in zip(test_preds, test_true)) / len(test_true)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("Classification report:")
    print(classification_report(test_true, test_preds,
                                target_names=["Male", "Female"], digits=4, zero_division=0))
    wandb.log({"test_accuracy": test_accuracy})
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

    # PA-100K layout: <root>/release_data/release_data/000001.jpg ... and <root>/annotation.mat
    images_dir = "/mnt/e/workspace/Dataset/PA-100K/release_data/release_data"
    annotation_mat_path = "/mnt/e/workspace/Dataset/PA-100K/annotation.mat"
    batch_size = 32

    GenderClassification(images_dir, annotation_mat_path, train_transform, val_transform,
                         batch_size, use_pretrained=True)
