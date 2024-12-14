import os
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

# Paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('./trained_model', size = (224,224))
model = ViTForImageClassification.from_pretrained('./trained_model')
model.to(device)

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs).logits
    predicted_class = outputs.argmax(-1).item()
    return predicted_class

# Test the prediction
image_path = './vt/test/stamp_2_1.png'  # Replace with your test image path
predicted_class = predict_image(image_path)
print(f"Predicted Class: {predicted_class}")
