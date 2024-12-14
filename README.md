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
   - Organize the collected dataset into `train/` and `val/` subdirectories under `./vt/`.
   - Ensure each subdirectory contains `good` and `bad` images for respective labels.
2. **Model Training**:
   - Run the `train.py` script to train the Vision Transformer model.
   - Processed data undergoes transformations like resizing and normalization to fit the model’s requirements.

## Key Components

### 1. CameraApp (File: `camera_app4.py`)

- **Initialization**: Sets up the camera, model, and GUI for capturing and displaying live camera feed.
- **Key Methods**:
  - `start_camera()`: Starts the camera and initializes real-time feed.
  - `update_frame()`: Continuously updates the feed on the GUI canvas.
  - `detect_defect(frame)`: Processes the frame to detect defects in individual stamps.
  - `crop_stamps(image_size)`: Defines the cropping dimensions for the six stamps based on predefined offsets and dimensions.
- **Dependencies**: `dvp` library for camera control, `torch` and `transformers` for model inference, `tkinter` for GUI.

#### Explanation of Code Snippet: Cropping Logic

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

- **Purpose**: Dynamically calculates the coordinates for cropping regions (stamps) from the left side of the image.
- **Logic**:
  - The horizontal coordinate `x` is adjusted by subtracting the product of column index and `STAMP_WIDTH` along with a `LEFT_OFFSET`.
  - The vertical coordinate `y` is set to 0 as the stamps are horizontally aligned.
  - The width and height are determined using predefined constants (`STAMP_WIDTH`, `STAMP_HEIGHT`).
  - Similar logic is applied to crop stamps on the right side, adjusted with a `RIGHT_OFFSET`.

### 2. ManualClassifier (File: `manual_classifier.py`)

- **Purpose**: Provides a GUI for users to manually classify images.
- **Key Features**:
  - Load images from a folder.
  - Display six images at a time with current classification.
  - Toggle classification and commit changes (renames files based on new labels).
  - Navigation for viewing more images.

### 3. Training Script (File: `train.py`)

- **Purpose**: Trains the Vision Transformer model on labeled data.
- **Workflow**:
  - Loads image datasets for training and validation.
  - Applies necessary transformations (resize, normalize).
  - Uses `torch` for training the ViT model with a cross-entropy loss.
  - Saves the trained model and feature extractor for deployment.

### Explanation of Code Snippet: Model Inference

```python
inputs = self.feature_extractor(images=stamp_image, return_tensors="pt")
inputs = {k: v.to(self.device) for k, v in inputs.items()}
outputs = self.model(**inputs).logits
predicted_class = outputs.argmax(-1).item()
```

- **Purpose**: Processes a single stamp to classify it as `good` or `bad`.
- **Steps**:
  1. **Feature Extraction**: Converts the image into a tensor compatible with the Vision Transformer model.
  2. **Inference**: Passes the tensor through the model to obtain logits (unnormalized predictions).
  3. **Classification**: The `argmax` function identifies the class with the highest probability.
  4. **Mapping**: The numeric class is mapped to labels (`good`/`bad`).

### 4. Navigation and GUI Workflow (File: `navigation.py`)

- **Purpose**: Provides a menu-based navigation system for the GUI and handles interactions between different application modules.

#### Workflow Logic:
1. **Main Menu**:
   - Displayed upon launching the application (`main_menu()` function).
   - Options to:
     - Start the defect detection interface.
     - Open the manual classification interface.
     - Exit the application.
2. **Switching Views**:
   - When a button is clicked, the corresponding interface (`CameraApp` or `ManualClassifier`) replaces the current GUI.
   - Widgets and frames of the current view are destroyed before transitioning to the new view.
3. **Defect Detection**:
   - Initiated when "Start Defect Detection" is selected.
   - Launches `CameraApp` to handle camera feed, image processing, and real-time defect detection.
4. **Manual Classification**:
   - Triggered when "Manual Classification" is selected.
   - Loads the `ManualClassifier` interface for reviewing and classifying images manually.
5. **Back Navigation**:
   - Provides a "Back" button in each interface to return to the main menu without restarting the application.

- **Key Features**:
  - Intuitive menu-based navigation.
  - Seamless transitions between views.
  - Unified control flow ensuring smooth user experience.

## Usage

### Running the Application

1. Ensure the camera is properly connected and configured.
2. Launch the application using the GUI (`gui.py`).
3. Use the main menu to:
   - Start defect detection.
   - Switch to manual classification.

### Training the Model

1. Prepare datasets with labeled images.
2. Place training and validation data in `./vt/train` and `./vt/val`, respectively.
3. Run the `train.py` script to fine-tune the model.

## Future Improvements

- Enhance cropping logic to dynamically synchronize with the alignment of the machine and camera, ensuring accurate stamp extraction even if the alignment changes.
- Integrate real-time performance metrics and alerts.
- Refine the manual classifier for better usability.

## Conclusion

This documentation provides a comprehensive overview of the Image Defect Detection System, detailing its components, logic, and usage. The system combines image processing, machine learning, and GUI-based workflows to deliver an efficient quality control solution.

