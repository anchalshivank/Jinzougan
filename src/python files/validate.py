import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from tqdm import tqdm

# Paths
val_dir = './vt/val'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('./vit_model')
model = ViTForImageClassification.from_pretrained('./vit_model')
model.to(device)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Load validation dataset
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Validation loop
def validate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

validate_model()
