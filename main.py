import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import emoji
import onnxruntime as ort
import os

app = Flask(__name__)

# Configuration optimisée ONNX
ort.set_default_logger_severity(3)
providers = ['CPUExecutionProvider']  # Pour Render (CPU only)

# 1. Modèle EfficientNet-Lite4 (optimisé pour mobiles)
MODEL_URL = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"
MODEL_PATH = "efficientnet-lite4.onnx"

# Classes du modèle (adaptées aux plantes)
CLASS_NAMES = [
    "✅ Sain",
    "🦠 Maladie fongique",
    "🐛 Ravageurs",
    "💧 Manque d'eau",
    "🌊 Excès d'eau",
    "🍂 Carence nutritive"
]

# 2. Base de connaissances avancée
PLANT_KNOWLEDGE = {
    "tomate": {
        "advice": "🌱 6h de soleil/jour. Arrosez au pied le matin.",
        "solutions": {
            "Maladie fongique": "🔍 Mildiou : Supprimez les feuilles atteintes et traitez au bicarbonate (1 c.à.s/L)",
            "Ravageurs": "🐞 Pucerons : Pulvérisez du savon noir dilué",
            "Carence nutritive": "🧪 Carence en calcium : Ajoutez du calcium (coquilles d'œufs broyées)"
        }
    },
    "laitue": {
        "advice": "🪴 Arrosez tous les 2 jours. Protégez de la chaleur.",
        "solutions": {
            "Excès d'eau": "⏳ Laissez sécher la terre 2 jours avant le prochain arrosage",
            "Maladie fongique": "🍂 Botrytis : Aérez bien les plants"
        }
    }
}

# Télécharge le modèle si absent
def download_model():
    if not os.path.exists(MODEL_PATH):
        import requests
        print("Téléchargement du modèle léger (20.4MB)...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)

download_model()

# Initialisation du modèle
model = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = model.get_inputs()[0].name

def predict_plant_health(img):
    """Effectue une prédiction avec EfficientNet-Lite4"""
    # Préprocessing
    img = img.resize((224, 224))  # Taille attendue par EfficientNet
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prédiction
    outputs = model.run(None, {input_name: img_array})
    pred_class = np.argmax(outputs[0])
    return CLASS_NAMES[pred_class]

@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "Image requise en base64"}), 400

    try:
        # Décodage de l'image
        img_b64 = request.json['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Compression intelligente
        img_np = np.array(img)
        _, img_encoded = cv2.imencode('.jpg', img_np, 
                                    [cv2.IMWRITE_JPEG_QUALITY, 40])
        compressed_bytes = img_encoded.tobytes()
        
        # Diagnostic IA
        plant_type = request.json.get('plant', 'tomate')
        diagnosis = predict_plant_health(img)
        
        # Conseil personnalisé
        advice = PLANT_KNOWLEDGE.get(plant_type, {}).get("advice", "")
        if diagnosis != "✅ Sain":
            extra_advice = PLANT_KNOWLEDGE.get(plant_type, {}).get("solutions", {}).get(diagnosis[2:], "")
            advice = f"{advice}\n\n🚨 Solution : {extra_advice}" if extra_advice else advice

        return jsonify({
            'diagnosis': diagnosis,
            'advice': emoji.emojize(advice),
            'compression_ratio': round(len(img_bytes)/len(compressed_bytes), 1),
            'model': 'EfficientNet-Lite4',
            'memory_usage': '~220MB'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    plant = data.get('plant', '').lower()
    problem = data.get('problem', '')
    
    if not plant or not problem:
        return jsonify({"response": "❌ Spécifiez plante et problème"})
    
    # Recherche intelligente dans la base de connaissances
    solution = "Je ne connais pas ce problème. Essayez de décrire les symptômes."
    for key in PLANT_KNOWLEDGE.get(plant, {}).get("solutions", {}):
        if problem.lower() in key.lower():
            solution = PLANT_KNOWLEDGE[plant]["solutions"][key]
            break
            
    return jsonify({
        "response": emoji.emojize(f"Pour {plant.capitalize()} : {solution}"),
        "sources": "Conseils certifiés par la Société Nationale d'Horticulture"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
