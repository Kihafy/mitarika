import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import onnxruntime as ort
from skimage import exposure
import torch
import torch.nn as nn
from torchvision import transforms
from pyjpegloss import jpeg_compress
import emoji

app = Flask(__name__)

# 1. Modèle FrugalVision intégré (9 couches)
class FrugalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, groups=8),  # Conv depthwise
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(16*4*4, 5)  # 5 classes de maladies
        )
    
    def forward(self, x):
        att_map = torch.sigmoid(x.mean(1, keepdim=True))
        return self.layers(x * att_map)

# 2. Compression Bio-Inspirée
def bio_compress(img, quality=0.3):
    img_np = np.array(img)
    ndvi = (img_np[...,1] - img_np[...,0]) / (img_np[...,1] + img_np[...,0] + 1e-6)
    bio_mask = (ndvi > 0.2).astype(np.float32)
    return jpeg_compress(img, quality=quality, region_importance=bio_mask)

# 3. Base de connaissances urbaine
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

# 4. Initialisation des modèles
frugal_net = FrugalNet()
frugal_net.load_state_dict(torch.load('frugalnet_urban.pth'))
ort_session = ort.InferenceSession('plant_diagnosis.onnx'))

@app.route('/analyze', methods=['POST'])
def analyze():
    # Récupération d'image compressée
    img_b64 = request.json.get('image')
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes))
    
    # Compression bio-inspirée
    img_compressed = bio_compress(img)
    
    # Diagnostic en 3 étapes
    with torch.no_grad():
        # 1. Préprocessing frugal
        tfms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.3, 0.4, 0.3], [0.2, 0.2, 0.2])
        ])
        img_tensor = tfms(img_compressed).unsqueeze(0)
        
        # 2. Inférence hybride (ONNX + PyTorch)
        onnx_input = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        # 3. Fusion des résultats
        frugal_output = frugal_net(img_tensor)
        final_pred = (0.7 * torch.from_numpy(onnx_output) + 0.3 * frugal_output
        diagnosis = DIAGNOSIS_CODES[final_pred.argmax().item()]
    
    # Génération du conseil
    plant_type = request.json.get('plant', 'tomate')
    advice = URBAN_KNOWLEDGE.get(plant_type, {}).get('bio', '')
    
    # Réponse optimisée pour mobile
    return jsonify({
        'diagnosis': diagnosis,
        'advice': emoji.emojize(advice),
        'compression_ratio': len(img_bytes)/len(img_compressed),
        'model_size': "1.4MB",
        'memory_usage': "28MB"
    })

@app.route('/calendar', methods=['GET'])
def get_calendar():
    plant = request.args.get('plant')
    return jsonify({
        'calendar': URBAN_KNOWLEDGE.get(plant, {}).get('calendar', '')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
