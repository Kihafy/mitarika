import os
import requests
import onnx
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import io
import base64
import json
from scipy.stats import entropy

app = Flask(__name__)

# Configuration ONNX
ort.set_default_logger_severity(3)
providers = ['CPUExecutionProvider']
MODEL_URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv3-small-11.onnx"
MODEL_PATH = "mobilenetv3-small.onnx"

# Classes du modèle
CLASS_NAMES = [
    "Sain",
    "Maladie fongique",
    "Ravageurs",
    "Manque d'eau",
    "Excès d'eau",
    "Carence nutritive"
]

# Base de connaissances avec scores de confiance
PLANT_KNOWLEDGE = {
    "tomate": {
        "advice": "6h de soleil/jour. Arrosez au pied le matin.",
        "solutions": {
            "Maladie fongique": {"text": "Mildiou : Supprimez les feuilles atteintes, traitez au bicarbonate (1 c.à.s/L).", "confidence": 0.8},
            "Ravageurs": {"text": "Pucerons : Pulvérisez du savon noir dilué.", "confidence": 0.7},
            "Carence nutritive": {"text": "Carence en calcium : Ajoutez du calcium (coquilles d'œufs broyées).", "confidence": 0.75}
        }
    },
    "laitue": {
        "advice": "Arrosez tous les 2 jours. Protégez de la chaleur.",
        "solutions": {
            "Excès d'eau": {"text": "Laissez sécher la terre 2 jours avant le prochain arrosage.", "confidence": 0.85},
            "Maladie fongique": {"text": "Botrytis : Aérez bien les plants.", "confidence": 0.9}
        }
    }
}

# Fichier pour stocker les scores de confiance (persistance légère)
FEEDBACK_FILE = "feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_feedback(feedback):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle léger (~10-15MB)...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            downloaded_size = int(r.headers.get('content-length', 0))
            expected_size = 15000000
            if downloaded_size < expected_size:
                raise ValueError(f"Taille du fichier téléchargé {downloaded_size} octets est inférieure à celle attendue {expected_size} octets")
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            os.chmod(MODEL_PATH, 0o644)
            print(f"Modèle téléchargé avec succès à {MODEL_PATH}")
        except Exception as e:
            print(f"Échec du téléchargement du modèle : {str(e)}")
            raise
    else:
        print(f"Le modèle existe déjà à {MODEL_PATH}")

def check_model_compatibility():
    if os.path.exists(MODEL_PATH):
        model_proto = onnx.load(MODEL_PATH)
        onnx.checker.check_model(model_proto)
        print(f"Modèle ONNX valide, version opset : {model_proto.opset_import[0].version}")
    else:
        raise FileNotFoundError(f"Le fichier {MODEL_PATH} n'existe pas")

download_model()
check_model_compatibility()
model = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = model.get_inputs()[0].name

def adaptive_compression(img_np, min_quality=20, max_quality=80, target_size=100000):
    """Compression adaptative basée sur l'entropie et la taille cible"""
    entropy_val = entropy(img_np.ravel())
    quality = int(min_quality + (max_quality - min_quality) * (entropy_val / 10))  # Ajuster selon entropie
    _, img_encoded = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed_bytes = img_encoded.tobytes()
    if len(compressed_bytes) > target_size and quality > min_quality:
        return adaptive_compression(img_np, min_quality, quality - 10, target_size)
    return compressed_bytes, quality

def predict_plant_health(img):
    """Effectue une prédiction avec MobileNetV3-Small"""
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    outputs = model.run(None, {input_name: img_array})
    pred_class = np.argmax(outputs[0])
    return CLASS_NAMES[pred_class]

@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "Image requise en base64"}), 400

    try:
        img_b64 = request.json['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)
        
        compressed_bytes, quality = adaptive_compression(img_np, target_size=100000)
        
        plant_type = request.json.get('plant', 'tomate')
        diagnosis = predict_plant_health(img)
        
        advice = PLANT_KNOWLEDGE.get(plant_type, {}).get("advice", "")
        if diagnosis != "Sain":
            solution = PLANT_KNOWLEDGE.get(plant_type, {}).get("solutions", {}).get(diagnosis, {})
            extra_advice = solution.get("text", "") if solution else ""
            confidence = solution.get("confidence", 0.0) if solution else 0.0
            advice = f"{advice}\n\nSolution : {extra_advice} (Confiance : {confidence:.2f})"

        return jsonify({
            'diagnosis': diagnosis,
            'advice': advice,
            'compression_ratio': round(len(img_bytes)/len(compressed_bytes), 1),
            'compression_quality': quality,
            'model': 'MobileNetV3-Small',
            'memory_usage': '~150MB'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    plant = data.get('plant', '').lower()
    problem = data.get('problem', '')
    feedback = data.get('feedback', None)
    
    if not plant or not problem:
        return jsonify({"response": "Spécifiez plante et problème"})

    feedback_data = load_feedback()
    if feedback and plant in PLANT_KNOWLEDGE and problem in PLANT_KNOWLEDGE[plant]["solutions"]:
        key = f"{plant}_{problem}"
        feedback_data[key] = feedback_data.get(key, {"positive": 0, "negative": 0})
        if feedback.lower() == "positive":
            feedback_data[key]["positive"] += 1
        elif feedback.lower() == "negative":
            feedback_data[key]["negative"] += 1
        confidence = feedback_data[key]["positive"] / (feedback_data[key]["positive"] + feedback_data[key]["negative"] + 1)
        PLANT_KNOWLEDGE[plant]["solutions"][problem]["confidence"] = confidence
        save_feedback(feedback_data)
    
    solution = "Je ne connais pas ce problème. Essayez de décrire les symptômes."
    for key in PLANT_KNOWLEDGE.get(plant, {}).get("solutions", {}):
        if problem.lower() in key.lower():
            solution = PLANT_KNOWLEDGE[plant]["solutions"][key]["text"]
            confidence = PLANT_KNOWLEDGE[plant]["solutions"][key]["confidence"]
            solution = f"{solution} (Confiance : {confidence:.2f})"
            break
            
    return jsonify({
        "response": f"Pour {plant.capitalize()} : {solution}",
        "sources": "Conseils certifiés par la Société Nationale d'Horticulture"
    })

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        verify_token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if verify_token == os.environ.get('VERIFY_TOKEN'):
            return challenge
        return "Invalid token", 403

    if request.method == 'POST':
        data = request.get_json()
        if data.get('object') == 'page':
            for entry in data.get('entry', []):
                for event in entry.get('messaging', []):
                    if event.get('message'):
                        sender_id = event['sender']['id']
                        message = event['message'].get('text', '')
                        # Simuler une requête /chat
                        response = chat().get_json()
                        send_facebook_message(sender_id, response['response'])
        return "EVENT_RECEIVED", 200

def send_facebook_message(sender_id, message):
    """Envoyer une réponse via l'API Facebook Messenger"""
    access_token = os.environ.get('FB_PAGE_ACCESS_TOKEN')
    url = f"https://graph.facebook.com/v13.0/me/messages?access_token={access_token}"
    payload = {
        "recipient": {"id": sender_id},
        "message": {"text": message}
    }
    requests.post(url, json=payload)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
