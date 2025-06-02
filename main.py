import os
import random
import gc
import onnx
from onnx import version_converter
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import numpy as np
from io import BytesIO
import requests
from dotenv import load_dotenv
import logging

# Charger les variables d'environnement
load_dotenv()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Pour Hugging Face

# Configuration pour 512 Mo
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI()

# Modèle ONNX
MODEL_PATH = "MobileNet.onnx"

# Réponses pré-définies
PREDEFINED_RESPONSES = [
    "Je détecte un objet intéressant!",
    "C'est probablement un élément du quotidien.",
    "Le modèle reconnaît un motif connu.",
]

def get_fallback_response():
    return random.choice(PREDEFINED_RESPONSES)

def load_onnx_model():
    """Charge le modèle ONNX optimisé"""
    try:
        model = onnx.load(MODEL_PATH)
        if model.opset_import[0].version > 19:
            model = version_converter.convert_version(model, 19)
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions()
        )
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur ONNX: {str(e)}")

model = load_onnx_model()

def preprocess_image(image: Image.Image):
    """Prétraitement optimisé"""
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)[:, :, :3]
    return np.expand_dims(img_array.transpose(2, 0, 1) / 255.0, axis=0)

async def query_huggingface(class_id: int, confidence: float):
    """Utilise l'API Hugging Face avec gestion d'erreur"""
    if not HF_API_KEY:
        return None
        
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/distilgpt2",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": f"Class #{class_id} ({confidence:.0%}):"},
            timeout=3
        )
        return response.json()[0]['generated_text'][:100]
    except:
        return None

@app.get("/")
async def root():
    return {"message": "API opérationnelle"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        input_tensor = preprocess_image(image)
        
        outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
        top_index = np.argmax(outputs[0])
        confidence = float(outputs[0][0, top_index])
        
        interpretation = await query_huggingface(top_index, confidence) or get_fallback_response()

        return JSONResponse({
            "prediction": int(top_index),
            "confidence": f"{confidence:.2%}",
            "interpretation": interpretation[:150]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()

# Intégration Messenger complète
@app.get("/webhook")
async def verify_webhook(request: Request):
    if request.query_params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(request.query_params.get("hub.challenge"))
    raise HTTPException(status_code=403)

@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        data = await request.json()
        entry = data.get("entry", [{}])[0]
        messaging = entry.get("messaging", [{}])[0]
        
        # Vérification des tokens
        if not PAGE_ACCESS_TOKEN or not VERIFY_TOKEN:
            raise HTTPException(status_code=500, detail="Tokens non configurés")
        
        sender_id = messaging.get("sender", {}).get("id")
        if not sender_id:
            return {"status": "ok"}
        
        # Gestion des messages
        if "message" in messaging:
            message = messaging["message"]
            
            # Réponse texte simple
            if "text" in message:
                response_text = "Envoyez-moi une photo pour analyse!"
            
            # Gestion des images
            elif "attachments" in message:
                attachment = message["attachments"][0]
                if attachment["type"] == "image":
                    image_url = attachment["payload"]["url"]
                    
                    # Téléchargement et traitement
                    try:
                        image_response = requests.get(image_url, timeout=5)
                        image = Image.open(BytesIO(image_response.content)).convert("RGB")
                        input_tensor = preprocess_image(image)
                        
                        outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
                        top_index = np.argmax(outputs[0])
                        confidence = float(outputs[0][0, top_index])
                        
                        interpretation = await query_huggingface(top_index, confidence) or get_fallback_response()
                        response_text = f"🔍 Résultat: Classe {top_index} ({confidence:.0%})\n💡 {interpretation[:100]}"
                    except Exception as e:
                        response_text = f"Erreur d'analyse: {str(e)[:50]}"
                    finally:
                        if 'image' in locals():
                            image.close()
                        gc.collect()
            
            # Envoi de la réponse à Messenger
            requests.post(
                f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}",
                json={
                    "recipient": {"id": sender_id},
                    "message": {"text": response_text}
                },
                timeout=3
            )
        
        return {"status": "ok"}
    except Exception as e:
        print(f"ERREUR WEBHOOK: {str(e)}")
        raise HTTPException(status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
