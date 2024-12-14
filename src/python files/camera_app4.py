from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from dvp import *  # Ensure correct camera SDK import
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

STAMP_HEIGHT = 101
STAMP_HEIGHT_LEFT_CORRECTION = 0
STAMP_HEIGHT_RIGHT_CORRECTION = 5
STAMP_WIDTH = 610
LEFT_OFFSET = 25
RIGHT_OFFSET = 105

class CameraApp:
    def __init__(self, parent, back_to_main):
        self.parent = parent
        self.running = False
        self.collect_samples = False
        self.dataset_dir = None
        self.camera = None

        # Load the model and feature extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('./trained_model', size = (224,224))
        self.model = ViTForImageClassification.from_pretrained('./trained_model')
        self.model.to(self.device)

        # Create a Canvas to display the video feed
        self.canvas = tk.Canvas(self.parent, width=800, height=200)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control buttons
        control_frame = ttk.Frame(self.parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = tk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = tk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.collect_samples_var = tk.BooleanVar()
        self.sample_checkbox = tk.Checkbutton(
            control_frame, text="Collect Sample Data", variable=self.collect_samples_var,
            command=self.toggle_sample_collection
        )
        self.sample_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

        self.back_button = tk.Button(control_frame, text="Back", command=self.back_to_main)
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.back_to_main_callback = back_to_main

    def toggle_sample_collection(self):
        """Enable or disable sample data collection."""
        self.collect_samples = self.collect_samples_var.get()
        if self.collect_samples:
            # Set up the dataset directory based on current time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_dir = os.path.join("dataset", f"dataset_{timestamp}")
            os.makedirs(self.dataset_dir, exist_ok=True)
            print(f"Dataset directory created: {self.dataset_dir}")
        else:
            self.dataset_dir = None
            print("Sample data collection disabled.")

    def start_camera(self):
        """Start the real camera feed."""
        try:
            self.camera = Camera(0)  # Use camera index 0
            conveyor_speed = 10  # Example conveyor speed
            self.configure_camera(conveyor_speed)
            self.camera.Start()
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_frame()
        except Exception as e:
            print(f"Failed to start camera: {e}")

    def stop_camera(self):
        """Stop the real camera feed."""
        if self.camera:
            try:
                self.camera.Stop()
            except Exception as e:
                print(f"Error stopping camera: {e}")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def configure_camera(self, conveyor_speed):
        """Configure camera settings dynamically."""
        try:
            line_rate = 10000
            self.camera.LineRate = line_rate
            print(f"Line rate set to: {line_rate} lines/sec")
        except Exception as e:
            print(f"Failed to set line rate: {e}")

        try:
            gain = 1.2
            self.camera.AnalogGain = gain
        except Exception as e:
            print(f"Failed to set the aalog gain {e}")
        try:
            roi = self.camera.Roi
            roi.W = 4000
            roi.H = 190
            self.camera.Roi = roi
            print(f"ROI set: X={roi.X}, Y={roi.Y}, W={roi.W}, H={roi.H}")
        except Exception as e:
            print(f"Failed to configure ROI: {e}")

        try:
            # exposure = max(600 / (conveyor_speed / 10), 100)
            self.camera.Exposure = 600
            print(f"Exposure set to: {self.camera.Exposure}")
        except Exception as e:
            print(f"Failed to set exposure: {e}")

        try:
            self.camera.TriggerState = False
            self.camera.TriggerSource = TriggerSource.TRIGGER_SOURCE_LINE3
        except Exception as e:
            print("Failed to configure hardware trigger:", e)

    def update_frame(self):
        """Update the video feed on the canvas."""
        if not self.running:
            return

        try:
            frame_buffer = self.camera.GetFrame(4000)
            frame = self.frame2mat(frame_buffer)
            
            if frame is not None:
                # Convert BGR to RGB for display
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame for display
                # frame = cv2.resize(frame, (1600, 400))

                # Optionally detect defects and save stamps
                if self.collect_samples and self.dataset_dir:
                    self.detect_defect(frame)

                # Convert to ImageTk for display
                image = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
                self.canvas.image = image  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Error capturing frame: {e}")

        self.parent.after(10, self.update_frame)

    def frame2mat(self, frame_buffer):
        """Convert frame buffer to a NumPy array."""
        try:
            frame, buffer = frame_buffer
            height = int(frame.iHeight)
            width = int(frame.iWidth)
            mat = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
            return mat
        except Exception as e:
            print(f"Error converting frame buffer to matrix: {e}")
            return None

    def detect_defect(self, frame):
        """Detect if stamps are good or bad and save them."""
        if not self.dataset_dir:
            return False

        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rectangles = self.crop_stamps(frame.shape)
        # print(rectangles)
        any_defective = False
        save_path = os.path.join(self.dataset_dir, f"{base_timestamp}_fame.jpg")
        result = cv2.imwrite(save_path, frame)
        for i, rect in enumerate(rectangles):
            try:
                x, y, w, h = rect
                x, y, w, h = int(x), int(y), int(w), int(h) 
                stamp_roi = frame[y:y + h, x:x + w]
            
                # Convert to PIL Image
                stamp_image = Image.fromarray(stamp_roi)
            
                # Resize image for ViT
                stamp_image = stamp_image.resize((224, 224))
                stamp_image = stamp_image.convert("RGB")
                
                # Preprocess and run inference
                inputs = self.feature_extractor(images=stamp_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs).logits
                predicted_class = outputs.argmax(-1).item()
                label = "good" if predicted_class == 1 else "bad"
                if label == "bad":
                    any_defective = True
                # Save stamp (no separate directories, all in dataset_dir)
                # Convert back to BGR for OpenCV saving if needed
                stamp_bgr = cv2.cvtColor(stamp_roi, cv2.COLOR_RGB2BGR)

                save_path = os.path.join(self.dataset_dir, f"{base_timestamp}_stamp{i}_{label}.jpg")
                result = cv2.imwrite(save_path, stamp_bgr)
                if result:
                    print(f"Stamp saved: {save_path} with label {label}")
                else:
                    print(f"Failed to save image at {save_path}")
            except Exception as e:
                print(f"Error processing stamp {i}: {e}")

        return any_defective

    def crop_stamps(self, image_size):
        """Crop stamps based on predefined sizes and offsets."""
        rectangles = []
        # Left side stamps
        for col in range(1, 4):
            rect = [
                max(0, image_size[1] // 2 - col * STAMP_WIDTH - LEFT_OFFSET),
                max(0, STAMP_HEIGHT * 0),
                STAMP_WIDTH,
                STAMP_HEIGHT + STAMP_HEIGHT_LEFT_CORRECTION,
            ]
            if rect[0] + rect[2] <= image_size[1] and rect[1] + rect[3] <= image_size[0]:
                rectangles.append(rect)

        # Right side stamps
        for col in range(3):
            rect = [
                max(0, image_size[1] // 2 + RIGHT_OFFSET + col * STAMP_WIDTH),
                max(0, (STAMP_HEIGHT + STAMP_HEIGHT_RIGHT_CORRECTION) * (2 - 0)),
                STAMP_WIDTH,
                STAMP_HEIGHT + STAMP_HEIGHT_RIGHT_CORRECTION,
            ]
            if rect[0] + rect[2] <= image_size[1] and rect[1] + rect[3] <= image_size[0]:
                rectangles.append(rect)

        return rectangles

    def back_to_main(self):
        """Go back to the main menu."""
        self.stop_camera()
        self.back_to_main_callback()
