# Documentation: Image Defect Detection System

## Overview

The Image Defect Detection System is designed to automate the process of capturing, analyzing, and classifying images for quality control. The workflow includes image acquisition via a camera, image cropping for stamp extraction, defect detection, and classification using a pre-trained Vision Transformer (ViT) model. A manual classification interface is also provided for human intervention when needed.

## System Workflow

### 1. Image Acquisition

- **Source**: A line-scan camera interfaced using the `dvp` library.
- **Resolution**: The camera captures images with a resolution of 4000x190 pixels.
- **Configuration**:
  - ROI (Region of Interest): Configured to match the resolution.
  - Line rate and analog gain are dynamically set.
  - Trigger source ensures synchronization with the hardware conveyor system.

### 2. Image Preprocessing

- The captured image is divided into six segments ("stamps") using predefined offsets and dimensions.
- Each segment is cropped and resized to match the input size required by the Vision Transformer model.

### 3. Defect Detection

- The system processes each stamp:
  1. Crops the region.
  2. Converts the region to RGB and resizes it to 224x224.
  3. Classifies the stamp as "good" or "bad" using the ViT model.
  4. Saves the stamp in a dataset directory with a label (`good`/`bad`) for future analysis.

### 4. Classification

- **Automated**: The Vision Transformer model predicts the class of each stamp.
- **Manual**: A separate interface allows users to manually classify and reclassify images.

## Folder Structure and Data Collection

### Folder Structure

- **Root Directory**: Contains all scripts and model-related files.
- **Dataset Directory**:
  - `dataset/`: Automatically created when enabling sample collection during defect detection.
  - Subfolders named with timestamps (e.g., `dataset_YYYYMMDD_HHMMSS`) to store captured images.
  - Each image is labeled as `good` or `bad` and saved with a unique name (e.g., `timestamp_stampX_good.jpg`).

### Data Collection Workflow

1. **Camera Capture**:
   - Images are saved during defect detection based on the current timestamp.
   - Full-frame and individual stamps are saved in the dataset directory.

2. **Manual Classification**:
   - Allows reclassification of images into `good` or `bad` categories.
   - Reclassified images are renamed and updated in the dataset folder.

### Data Usage for Training
1. **Preparation**:
   - After manual classification, organize the dataset into `train/` and `val/` subdirectories under a directory such as `./vt/`.
   - Ensure each subdirectory (`train` and `val`) contains two folders: `good` and `bad`, each with their respective images.

2. **Model Training**:
   - Run the training script (`train.py`) to fine-tune the Vision Transformer model.
   - Images undergo transformations like resizing and normalization to fit the model’s requirements.

## Key Components

### 1. CameraApp (File: `camera_app4.py`)

- **Initialization**: Sets up the camera, model, and GUI for capturing and displaying the live camera feed.
- **Key Methods**:
  - `start_camera()`: Starts the camera and initializes real-time feed.
  - `update_frame()`: Continuously updates the feed on the GUI canvas.
  - `detect_defect(frame)`: Processes the frame to detect defects in individual stamps.
  - `crop_stamps(image_size)`: Defines the cropping dimensions for the six stamps based on predefined offsets and dimensions.
- **Dependencies**: `dvp` library for camera control, `torch` and `transformers` for model inference, `tkinter` for GUI.

#### Example: Cropping Logic

```python
for col in range(1, 4):
    rect = [
        max(0, image_size[1] // 2 - col * STAMP_WIDTH - LEFT_OFFSET),
        max(0, STAMP_HEIGHT * 0),
        STAMP_WIDTH,
        STAMP_HEIGHT + STAMP_HEIGHT_LEFT_CORRECTION,
    ]
    rectangles.append(rect)
```

- Dynamically calculates coordinates to crop stamps from the left side of the image. Similar logic is applied for the right side using `RIGHT_OFFSET`.

### 2. ManualClassifier (File: `manual_classifier.py`)

- **Purpose**: GUI interface for manual classification.
- **Features**:
  - Load images from a folder.
  - Display six images at a time with their current classification.
  - Toggle classification and commit changes (renaming files accordingly).
  - Navigate through multiple sets of images.

### 3. Model Training, Validation, and Testing (File: `train.py`)

**Purpose**: To train, validate, and test the ViT model on your labeled data.

**Before Training**:
- Prepare your dataset:  
  - After using the `ManualClassifier`, separate your images into `good` and `bad` folders.
  - Organize them as:
    ```
    vt/
      train/
        good/
          image1_good.jpg
          ...
        bad/
          image2_bad.jpg
          ...
      val/
        good/
          val_image1_good.jpg
          ...
        bad/
          val_image2_bad.jpg
          ...
    ```
    (Optionally, create a `test` directory with the same structure for final evaluation.)

**Training Script Overview**:
- Loads the pre-trained ViT model (`google/vit-base-patch16-224-in21k`) and customizes the output layer based on the number of classes.
- Applies data transformations (resize, normalize).
- Trains for a specified number of epochs, computing training loss and validating on the validation set after each epoch.

**Validation**:
- Conducted after each training epoch.
- Evaluates model performance on the validation set, reporting validation loss and accuracy.
- Useful for early stopping or hyperparameter tuning.

**Testing**:
- Conducted after training completion on the test set (if provided).
- Provides a final measure of model performance on unseen data.

**Code Example**:

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.optim import AdamW
from tqdm import tqdm

# Paths (adjust as necessary)
train_dir = './vt/train'
val_dir = './vt/val'
test_dir = './vt/test'  # Include this if you have a test set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)  # if test set is available

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # if test set is available

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(train_dataset.classes)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
num_epochs = 5

def train_model():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        validate_model()

def validate_model():
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    model.train()

def test_model():
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    model.train()

# Run training and validation
train_model()

# After training completes, optionally test if a test set is provided
test_model()

# Save the model and feature extractor
model.save_pretrained('./vit_model')
feature_extractor.save_pretrained('./vit_model')
print("Model training, validation, and testing complete. Model saved to ./vit_model.")
```

### Explanation: Model Inference Example

```python
inputs = self.feature_extractor(images=stamp_image, return_tensors="pt")
inputs = {k: v.to(self.device) for k, v in inputs.items()}
outputs = self.model(**inputs).logits
predicted_class = outputs.argmax(-1).item()
```

- Converts the image into a tensor.
- Performs inference on the ViT model.
- Chooses the class with the highest probability as the final prediction.

### 4. Navigation and GUI Workflow (File: `navigation.py`)

- **Purpose**: Provides a menu-based navigation system for the GUI and handles transitions between `CameraApp` and `ManualClassifier`.
- **Features**:
  - Main menu for starting defect detection or opening manual classification.
  - Back button for returning to the main menu.

## Usage

### Running the Application

1. Ensure the camera is properly connected and configured.
2. Launch the application using the GUI (`gui.py`).
3. Use the main menu to:
   - Start defect detection.
   - Switch to manual classification.

### Training the Model

1. Prepare datasets with labeled images (`good` and `bad`).
2. Organize them under `./vt/train` and `./vt/val` (and optionally `./vt/test`).
3. Run `train.py` to train the ViT model. Validation and testing steps are included.

## Future Improvements

- Enhance cropping logic to dynamically adapt to machine alignment changes.
- Add real-time performance metrics and alerts.
- Improve manual classification tool usability and feature set.

## Conclusion

This documentation provides a comprehensive overview of the Image Defect Detection System and its components. It outlines the process from image acquisition to automated and manual classification, and details how to train, validate, and test a Vision Transformer model for high-accuracy image defect detection in industrial quality control scenarios.

---
