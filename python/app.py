# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import os

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load a pretrained ResNet model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=True)
# model = model.to(device)
# model.eval()

# # Image Preprocessing Function
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0).to(device)

# # Compute damage percentage using deep learning
# def calculate_damage_percentage(original, damaged):
#     original_tensor = preprocess_image(original)
#     damaged_tensor = preprocess_image(damaged)

#     with torch.no_grad():
#         original_features = model(original_tensor).squeeze().cpu().numpy()
#         damaged_features = model(damaged_tensor).squeeze().cpu().numpy()

#     # Compute similarity and convert to damage percentage
#     similarity = np.dot(original_features, damaged_features) / (
#         np.linalg.norm(original_features) * np.linalg.norm(damaged_features)
#     )
#     damage_percentage = (1 - similarity) * 100
#     damage_percentage = max(0, min(damage_percentage, 100))

#     return round(float(damage_percentage), 2)  # Convert NumPy float to Python float

# @app.route('/compare-images', methods=['POST'])
# def compare_images():
#     try:
#         if 'image1' not in request.files or 'image2' not in request.files:
#             print("🚨 Error: One or both images are missing in request.files")
#             return jsonify({"error": "Both images must be uploaded"}), 400

#         file1 = request.files['image1']
#         file2 = request.files['image2']

#         img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#         img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

#         if img1 is None or img2 is None:
#             print("🚨 Error: Invalid image data received.")
#             return jsonify({"error": "Invalid image data"}), 400

#         damage_percentage = calculate_damage_percentage(img1, img2)

#         return jsonify({
#             "damagePercentage": float(damage_percentage)  # Convert to standard float
#         })
#     except Exception as e:
#         print(f"🔥 Flask Error: {str(e)}")  # Print error in Flask console
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)



# //////////////////////////////////////////////////////////////////////
# # Siamese Network Implement
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# import os

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # ✅ Define Siamese Network
# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # Output 128 features

#     def forward(self, img1, img2):
#         feat1 = self.resnet(img1)
#         feat2 = self.resnet(img2)
#         return feat1, feat2

# # ✅ Load Siamese Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SiameseNetwork().to(device)
# model.eval()

# # ✅ Image Preprocessing Function
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0).to(device)

# # ✅ Compute Damage Percentage Using Siamese Network
# def calculate_damage_percentage(original, damaged):
#     original_tensor = preprocess_image(original)
#     damaged_tensor = preprocess_image(damaged)

#     with torch.no_grad():
#         feat1, feat2 = model(original_tensor, damaged_tensor)

#     # Compute Euclidean Distance between Feature Vectors
#     distance = torch.norm(feat1 - feat2, p=2).item()

#     # Convert Distance to Damage Percentage (Higher Distance = More Damage)
#     damage_percentage = min(max(distance * 10, 0), 100)  # Scale factor to map to 0-100%

#     return round(float(damage_percentage), 2)  # Convert to standard float

# # ✅ API Endpoint to Compare Images
# @app.route('/compare-images', methods=['POST'])
# def compare_images():
#     try:
#         if 'image1' not in request.files or 'image2' not in request.files:
#             print("🚨 Error: One or both images are missing in request.files")
#             return jsonify({"error": "Both images must be uploaded"}), 400

#         file1 = request.files['image1']
#         file2 = request.files['image2']

#         img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#         img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

#         if img1 is None or img2 is None:
#             print("🚨 Error: Invalid image data received.")
#             return jsonify({"error": "Invalid image data"}), 400

#         damage_percentage = calculate_damage_percentage(img1, img2)

#         return jsonify({
#             "damagePercentage": float(damage_percentage)  # Convert NumPy float to Python float
#         })
#     except Exception as e:
#         print(f"🔥 Flask Error: {str(e)}")  # Print error in Flask console
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)


#  //////////////////////////////////////////////////////////////////////////
#  U NET Implement

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Siamese Network (Unchanged)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # Output 128 features

    def forward(self, img1, img2):
        feat1 = self.resnet(img1)
        feat2 = self.resnet(img2)
        return feat1, feat2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siamese_model = SiameseNetwork().to(device)
siamese_model.eval()

# ✅ U-Net Model for Damage Segmentation (New)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove fully connected layer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=1),  # Output 1-channel mask
            nn.Sigmoid()  # Normalize output (0-1 range)
        )

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.decoder(x)
        return x

# ✅ Load Pretrained U-Net Model
unet_model = UNet().to(device)
unet_model.eval()

# ✅ Image Preprocessing Function (Unchanged)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# ✅ Compute Damage Percentage Using Siamese Network (Unchanged)
def calculate_damage_percentage(original, damaged):
    original_tensor = preprocess_image(original)
    damaged_tensor = preprocess_image(damaged)

    with torch.no_grad():
        feat1, feat2 = siamese_model(original_tensor, damaged_tensor)

    distance = torch.norm(feat1 - feat2, p=2).item()
    damage_percentage = min(max(distance * 10, 0), 100)  # Scale factor to map to 0-100%
    return round(float(damage_percentage), 2)

# ✅ New Function: Segment Damage Using U-Net
def segment_damage(image):
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        mask = unet_model(image_tensor)
    
    mask = mask.squeeze().cpu().numpy()  # Convert to NumPy
    mask = (mask > 0.5).astype(np.uint8)  # Thresholding (1 = damaged, 0 = normal)

    # ✅ Calculate Damage Area Percentage
    total_pixels = mask.size
    damaged_pixels = np.sum(mask)
    damage_area_percentage = (damaged_pixels / total_pixels) * 100

    return mask, round(float(damage_area_percentage), 2)

# ✅ API Endpoint for Siamese Network Image Comparison (Unchanged)
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
            "damagePercentage": float(damage_percentage)  # Convert NumPy float to Python float
        })
    except Exception as e:
        print(f"🔥 Flask Error: {str(e)}")  # Print error in Flask console
        return jsonify({"error": str(e)}), 500

# ✅ New API Endpoint: Segment Damage Using U-Net
@app.route('/segment-damage', methods=['POST'])
def segment_damage_api():
    try:
        if 'image' not in request.files:
            print("🚨 Error: No image found in request")
            return jsonify({"error": "Image must be uploaded"}), 400

        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print("🚨 Error: Invalid image data received.")
            return jsonify({"error": "Invalid image data"}), 400

        mask, damage_area_percentage = segment_damage(img)

        return jsonify({
            "damageAreaPercentage": float(damage_area_percentage)  # Convert NumPy float to Python float
        })
    except Exception as e:
        print(f"🔥 Flask Error: {str(e)}")  # Print error in Flask console
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
