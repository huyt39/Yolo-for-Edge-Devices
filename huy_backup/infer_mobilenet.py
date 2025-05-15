import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from select_features import extract_features

# Custom dataset that reads images from subfolders and assigns gender labels from the dictionary.
class GenderDataset(Dataset):
	def __init__(self, data_dir, annotation_path, transform):
		"""
		:param data_dir: Directory containing subfolders for each identity.
		:param annotation_path: Directory where annotation files are stored.
		:param transform: Image transformations.
		"""
		self.samples = []
		self.transform = transform

		# Find annotation file corresponding to data_dir
		folder_name = os.path.basename(data_dir)
		annotation_file = os.path.join(annotation_path, folder_name + ".txt")

		if not os.path.exists(annotation_file):
			raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

		# Extract {ID: gender} mapping from annotation file
		selected_features = [1, 4]  # ID and Gender
		labels_dict = extract_features([annotation_file], selected_features)

		# Walk through each subfolder in data_dir.
		for id in os.listdir(data_dir):
			if id == "-1":
				continue

			subfolder = os.path.join(data_dir, id)
			if os.path.isdir(subfolder):
				# Use the subfolder name (id) to get the gender label.
				# Make sure keys in labels_dict are of the same type as 'id' (e.g., string)
				gender = labels_dict.get(id)
				if (gender == 2):
					# Optionally: skip if the id is not found in the dictionary.
					continue
					
				# Loop through each image file in the subfolder.
				for file in os.listdir(subfolder):
					if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
						img_path = os.path.join(subfolder, file)
						self.samples.append((img_path, gender))

	def __getitem__(self, index):
		path, gender = self.samples[index]
		image = Image.open(path).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image, gender

	def __len__(self):
		return len(self.samples)

def infer_gender(data_dir, annotation_path, transform, batch_size):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Create dataset and DataLoader.
	image_dataset = GenderDataset(data_dir, annotation_path, transform)
	image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
	
	# Load MobileNetV2 model.
	model = models.mobilenet_v2(pretrained=False)
	num_ftrs = model.classifier[1].in_features
	model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification (Male/Female)

	# Load trained weights
	model_path = "best_model_mobilenet.pth"
	if os.path.exists(model_path):
		try:
			state_dict = torch.load(model_path, map_location=device)
			model.load_state_dict(state_dict, strict=False)
			print("Model loaded successfully!")
		except Exception as e:
			print(f"Error loading model: {e}")
	else:
		print(f"Model file '{model_path}' not found!")
		return
	
	model.to(device)
	model.eval()

	correct = 0
	total = 0

	# Inference loop.
	with torch.no_grad():
		for images, labels in image_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			# Get predicted class (0 or 1)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total if total > 0 else 0
	print('Accuracy: {:.2f}%'.format(accuracy))
	return accuracy

# Example usage:
if __name__ == '__main__':
	# Define your transformations (example: resize and normalization).
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
	])

	base_data_dir = "/mnt/e/workspace/Dataset/P-DESTR/rois/jpg_Extracted_PIDS"
	annotation_path = "/mnt/e/workspace/Dataset/P-DESTR/dataset/P-DESTRE/annotation"
	batch_size = 32
	
	# Iterate through all subdirectories in base_data_dir
	for folder_name in os.listdir(base_data_dir):
		data_dir = os.path.join(base_data_dir, folder_name)
		if os.path.isdir(data_dir):  # Ensure it's a directory
			print(f"Running inference on folder: {folder_name}")
			infer_gender(data_dir, annotation_path, transform, batch_size)
