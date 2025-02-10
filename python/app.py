from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load a pretrained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Compute damage percentage using deep learning
def calculate_damage_percentage(original, damaged):
    original_tensor = preprocess_image(original)
    damaged_tensor = preprocess_image(damaged)

    with torch.no_grad():
        original_features = model(original_tensor).squeeze().cpu().numpy()
        damaged_features = model(damaged_tensor).squeeze().cpu().numpy()

    # Compute similarity and convert to damage percentage
    similarity = np.dot(original_features, damaged_features) / (
        np.linalg.norm(original_features) * np.linalg.norm(damaged_features)
    )
    damage_percentage = (1 - similarity) * 100
    damage_percentage = max(0, min(damage_percentage, 100))

    return round(float(damage_percentage), 2)  # Convert NumPy float to Python float

@app.route('/compare-images', methods=['POST'])
def compare_images():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            print("🚨 Error: One or both images are missing in request.files")
            return jsonify({"error": "Both images must be uploaded"}), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            print("🚨 Error: Invalid image data received.")
            return jsonify({"error": "Invalid image data"}), 400

        damage_percentage = calculate_damage_percentage(img1, img2)

        return jsonify({
            "damagePercentage": float(damage_percentage)  # Convert to standard float
        })
    except Exception as e:
        print(f"🔥 Flask Error: {str(e)}")  # Print error in Flask console
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
