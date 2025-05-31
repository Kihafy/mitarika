import os
import requests
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import json
from scipy.stats import entropy

app = Flask(__name__)

# Configuration ONNX
ort.set_default_logger_severity(3)
providers = ['CPUExecutionProvider']
MODEL_URL = "https://github.com/Kihafy/mitarika/releases/download/v1.0/mobilenetv3_rw_Opset17.onnx"
MODEL_PATH = "model.onnx"  # Nom plus court

# Classes du modèle
CLASS_NAMES = [
    "Sain",
    "Maladie fongique",
    "Ravageurs",
    "Manque d'eau",
    "Excès d'eau",
    "Carence nutritive"
]

# Base de connaissances optimisée
PLANT_KNOWLEDGE = {
    "tomate": {
        "advice": "6h de soleil/jour. Arrosez au pied le matin.",
        "solutions": {
            "Maladie fongique": "Mildiou : Supprimez les feuilles atteintes, traitez au bicarbonate (1 c.à.s/L)",
            "Ravageurs": "Pucerons : Pulvérisez du savon noir dilué",
            "Carence nutritive": "Carence en calcium : Ajoutez du calcium (coquilles d'œufs broyées)"
        }
    },
    "laitue": {
        "advice": "Arrosez tous les 2 jours. Protégez de la chaleur.",
        "solutions": {
            "Excès d'eau": "Laissez sécher la terre 2 jours avant le prochain arrosage",
            "Maladie fongique": "Botrytis : Aérez bien les plants"
        }
    }
}

def load_model():
    """Charge le modèle avec vérification robuste"""
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle...")
        try:
            with requests.get(MODEL_URL, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            if os.path.getsize(MODEL_PATH) < 5_000_000:
                raise ValueError("Fichier modèle trop petit")
                
        except Exception as e:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            raise RuntimeError(f"Échec téléchargement: {str(e)}")

    try:
        return ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception as e:
        raise RuntimeError(f"Erreur ONNX: {str(e)}")

# Initialisation
try:
    model = load_model()
    input_name = model.get_inputs()[0].name
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"⚠️ Mode dégradé: {str(e)}")
    model = None

def compress_image(img_np, target_size=100000):
    """Compression d'image optimisée"""
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    
    # Compression adaptative
    quality = 80
    while quality > 20:
        buffer.seek(0)
        img_pil.save(buffer, format='JPEG', quality=quality)
        if buffer.tell() <= target_size:
            break
        quality -= 10
    
    return buffer.getvalue(), quality

def predict_plant_health(img):
    """Prédiction optimisée"""
    if model is None:
        return "Erreur: Modèle non disponible"
    
    try:
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        outputs = model.run(None, {input_name: img_array})
        return CLASS_NAMES[np.argmax(outputs[0])]
    except Exception as e:
        print(f"Erreur prédiction: {str(e)}")
        return "Erreur d'analyse"

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint optimisé"""
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "Image requise en base64"}), 400

    try:
        # Décodage de l'image
        img_data = base64.b64decode(request.json['image'])
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
        
        # Compression
        compressed_img, quality = compress_image(img_np)
        
        # Diagnostic
        plant_type = request.json.get('plant', 'tomate').lower()
        diagnosis = predict_plant_health(img)
        
        # Conseil personnalisé
        advice = PLANT_KNOWLEDGE.get(plant_type, {}).get("advice", "")
        if diagnosis != "Sain":
            solution = PLANT_KNOWLEDGE.get(plant_type, {}).get("solutions", {}).get(diagnosis, "")
            if solution:
                advice = f"{advice}\n\nSolution recommandée : {solution}"

        return jsonify({
            'diagnosis': diagnosis,
            'advice': advice,
            'compression_quality': quality,
            'model_status': 'active' if model else 'degraded'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint simplifié pour le chat"""
    data = request.get_json()
    plant = data.get('plant', '').lower()
    problem = data.get('problem', '')
    
    if not plant or not problem:
        return jsonify({"response": "Spécifiez plante et problème"})

    solution = PLANT_KNOWLEDGE.get(plant, {}).get("solutions", {}).get(problem, 
              "Je ne connais pas ce problème. Consultez un spécialiste.")
    
    return jsonify({
        "response": f"Pour {plant.capitalize()} : {solution}",
        "source": "Base de connaissances horticoles"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
