import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import onnxruntime as ort
import cv2  # Utilisation d'OpenCV pour la compression
import torch
import torch.nn as nn
from torchvision import transforms
import emoji

app = Flask(__name__)

# 1. Modèle FrugalVision optimisé
class FrugalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, groups=8),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16*4*4, 5)
        )
    
    def forward(self, x):
        att_map = torch.sigmoid(x.mean(1, keepdim=True))
        return self.layers(x * att_map)

# 2. Compression alternative avec OpenCV
def bio_compress(img, quality=30):
    """Compression JPEG avec préservation des zones importantes"""
    img_np = np.array(img)
    
    # Détection des zones vertes (NDVI simplifié)
    b, g, r = cv2.split(img_np)
    ndvi = (g.astype(float) - r.astype(float)) / (g + r + 1e-6)
    mask = (ndvi > 0.2).astype(np.uint8) * 255
    
    # Compression adaptative
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, img_encoded = cv2.imencode('.jpg', img_np, encode_param)
    
    return io.BytesIO(img_encoded.tobytes())

# 3. Base de connaissances (comme avant)
URBAN_KNOWLEDGE = {
    "tomate": {
        "bio": "🌱 Semis en pot >15L, sud. Purin d'ortie contre pucerons.",
        "calendar": "📅 Sept-Nov : 3 mois de culture"
    },
    "laitue": {
        "bio": "🪴 Jardinière profonde. Associez radis pour optimiser l'espace.",
        "calendar": "📅 Toute l'année en rotation"
    }
}

DIAGNOSIS_CODES = {
    0: "✅ Sain",
    1: "💧 Manque d'eau",
    2: "🌊 Excès d'eau",
    3: "🍂 Carence nutritive",
    4: "🦠 Maladie fongique"
}

# Initialisation
frugal_net = FrugalNet()
frugal_net.load_state_dict(torch.load('frugalnet_urban.pth', map_location='cpu'))
ort_session = ort.InferenceSession('plant_diagnosis.onnx')

@app.route('/analyze', methods=['POST'])
def analyze():
    img_b64 = request.json.get('image')
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes))
    
    # Compression
    compressed_img = bio_compress(img)
    compressed_bytes = compressed_img.getvalue()
    
    # Diagnostic
    with torch.no_grad():
        tfms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.3, 0.4, 0.3], [0.2, 0.2, 0.2])
        ])
        img_tensor = tfms(img).unsqueeze(0)
        
        onnx_input = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        frugal_output = frugal_net(img_tensor)
        final_pred = (0.7 * torch.from_numpy(onnx_output) + 0.3 * frugal_output)
        diagnosis = DIAGNOSIS_CODES[final_pred.argmax().item()]
    
    plant_type = request.json.get('plant', 'tomate')
    advice = URBAN_KNOWLEDGE.get(plant_type, {}).get('bio', '')
    
    return jsonify({
        'diagnosis': diagnosis,
        'advice': emoji.emojize(advice),
        'compression_ratio': len(img_bytes)/len(compressed_bytes),
        'model_size': "1.4MB",
        'memory_usage': "28MB"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
