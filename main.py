# main.py
import os
import random
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import onnxruntime as ort

app = FastAPI()

MODEL_PATH = "MobileNet.onnx"
WELCOME_MESSAGES = [
    "👋 Bonjour ! Prêt à découvrir ce que je vois ?",
    "Salut ! Envoyez-moi une image et je vous dis ce qu'elle contient 📷.",
    "Bienvenue ! Je suis un assistant visuel intelligent. Que puis-je faire pour vous ?",
]

def get_welcome_message():
    return random.choice(WELCOME_MESSAGES)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Modèle introuvable. Placez 'MobileNet-v3-Small_w8a16.onnx' dans le dossier du projet.")
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        print("✅ Modèle ONNX chargé avec succès.")
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur ONNX : {e}")

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.transpose(2, 0, 1) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def welcome():
    return {"message": get_welcome_message()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_tensor})
        predictions = outputs[0][0]
        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index])

        return JSONResponse({
            "prediction": f"Classe #{top_index}",
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")
