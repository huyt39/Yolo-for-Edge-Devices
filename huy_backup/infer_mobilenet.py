import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report
from scipy.io import loadmat


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


def infer_gender(images_dir, annotation_mat_path, transform, batch_size, model_path, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PA100KGenderDataset(images_dir, annotation_mat_path, split, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Inference on PA-100K {split}: {len(dataset)} samples")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    model.to(device)
    model.eval()

    preds, true = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().tolist())
            true.extend(labels.cpu().tolist())

    accuracy = 100.0 * sum(p == t for p, t in zip(preds, true)) / len(true)
    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification report:")
    print(classification_report(true, preds, target_names=["Male", "Female"], digits=4, zero_division=0))
    return accuracy


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # PA-100K layout: <root>/release_data/release_data/000001.jpg ... and <root>/annotation.mat
    images_dir = "/mnt/e/workspace/Dataset/PA-100K/release_data/release_data"
    annotation_mat_path = "/mnt/e/workspace/Dataset/PA-100K/annotation.mat"
    model_path = "best_model_mobilenet.pth"
    batch_size = 32

    infer_gender(images_dir, annotation_mat_path, transform, batch_size, model_path, split="test")
