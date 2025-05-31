import os
import random
import onnx
from onnx import version_converter
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI()

MODEL_PATH_ORIGINAL = "MobileNet.onnx"
MODEL_PATH_CONVERTED = "MobileNet_19.onnx"
WELCOME_MESSAGES = [
    "👋 Bonjour ! Prêt à découvrir ce que je vois ?",
    "Salut ! Envoyez-moi une image et je vous dis ce qu'elle contient 📷.",
    "Bienvenue ! Je suis un assistant visuel intelligent. Que puis-je faire pour vous ?",
]

def get_welcome_message():
    return random.choice(WELCOME_MESSAGES)

def check_and_convert_model():
    """Vérifie l'opset du modèle et le convertit en opset 19 si nécessaire."""
    if not os.path.exists(MODEL_PATH_ORIGINAL):
        raise FileNotFoundError(f"❌ Modèle introuvable à {MODEL_PATH_ORIGINAL}. Placez 'MobileNet.onnx' dans le dossier du projet.")

    # Charger le modèle pour vérifier l'opset
    model = onnx.load(MODEL_PATH_ORIGINAL)
    opset_version = model.opset_import[0].version
    print(f"Version opset du modèle : {opset_version}")

    # Si l'opset est supérieur à 19, convertir vers opset 19
    if opset_version > 19:
        print("⚠️ Opset non supporté (>19). Conversion vers opset 19...")
        try:
            converted_model = version_converter.convert_version(model, 19)
            onnx.save(converted_model, MODEL_PATH_CONVERTED)
            onnx.checker.check_model(converted_model)
            print(f"✅ Modèle converti avec succès et sauvegardé à {MODEL_PATH_CONVERTED}.")
            return MODEL_PATH_CONVERTED
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la conversion du modèle : {e}")
    else:
        print(f"✅ Modèle compatible (opset {opset_version}). Aucun besoin de conversion.")
        return MODEL_PATH_ORIGINAL

def load_model():
    """Charge le modèle ONNX, en convertissant si nécessaire."""
    try:
        # Vérifier et convertir le modèle si nécessaire
        model_path = check_and_convert_model()
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✅ Modèle ONNX chargé avec succès depuis {model_path}.")
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur ONNX : {e}")

# Charger le modèle au démarrage
model = load_model()

def preprocess_image(image: Image.Image):
    """Prétraite l'image pour l'inférence."""
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.transpose(2, 0, 1) / 255.0  # Normalisation à [0,1]
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
        predictions = outputs[0]  # Corrigé : suppression de "0emn0"
        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index])

        return JSONResponse({
            "prediction": f"Classe #{top_index}",
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")
