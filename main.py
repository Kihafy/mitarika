import os
import random
import onnx
from onnx import version_converter
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import hmac
import hashlib
from io import BytesIO
import requests  # Assurez-vous d'avoir importé requests

app = FastAPI()

MODEL_PATH_ORIGINAL = "MobileNet.onnx"
MODEL_PATH_CONVERTED = "MobileNet_ops19.onnx"
WELCOME_MESSAGES = [
    "👋 Bonjour ! Prêt à découvrir ce que je vois ?",
    "Salut ! Envoyez-moi une image et je vous dis ce qu'elle contient 📷.",
    "Bienvenue ! Je suis un assistant visuel intelligent. Que puis-je faire pour vous ?",
]

# Récupérer les variables d'environnement depuis Render
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", )  # Récupère le token de vérification
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", )  # Token d'accès pour Messenger

def get_welcome_message():
    return random.choice(WELCOME_MESSAGES)

def check_and_convert_model():
    """Vérifie l'opset du modèle et le convertit en opset 19 si nécessaire."""
    if not os.path.exists(MODEL_PATH_ORIGINAL):
        raise FileNotFoundError(f"❌ Modèle introuvable à {MODEL_PATH_ORIGINAL}. Placez 'MobileNet.onnx' dans le dossier du projet.")

    model = onnx.load(MODEL_PATH_ORIGINAL)
    opset_version = model.opset_import[0].version
    print(f"Version opset du modèle : {opset_version}")

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
        model_path = check_and_convert_model()
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"✅ Modèle ONNX chargé avec succès depuis {model_path}.")
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur ONNX : {e}")

model = load_model()

def preprocess_image(image: Image.Image):
    """Prétraite l'image pour l'inférence."""
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.transpose(2, 0, 1) / 255.0
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
        predictions = outputs[0]
        top_index = int(np.argmax(predictions))
        confidence = float(predictions[top_index])

        return JSONResponse({
            "prediction": f"Classe #{top_index}",
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

@app.get("/webhook")
async def webhook_verify(request: Request):
    """Vérifie le webhook pour Messenger."""
    query = request.query_params
    mode = query.get("hub.mode")
    token = query.get("hub.verify_token")
    challenge = query.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("Webhook vérifié avec succès !")
        return int(challenge)  # Renvoie la challenge pour valider le webhook
    else:
        raise HTTPException(status_code=403, detail="Token de vérification incorrect")
@app.post("/webhook")
async def webhook(request: Request):
    """Gère les messages entrants de Messenger (texte et images)."""
    try:
        body = await request.json()
        print(f"Reçu : {body}")  # Log du corps de la requête
        if "entry" in body and len(body["entry"]) > 0:
            messaging = body["entry"][0].get("messaging", [])
            for event in messaging:
                # Ignorer les messages "echo" pour éviter une boucle infinie
                if "message" in event and event["message"].get("is_echo", False):
                    print("Message echo ignoré")
                    continue

                sender_id = event["sender"]["id"]
                response_message = {"recipient": {"id": sender_id}, "message": {"text": ""}}

                if "message" in event:
                    if "text" in event["message"]:
                        # Gérer les messages texte
                        user_text = event["message"]["text"]
                        print(f"Message texte reçu : {user_text}")
                        if user_text.lower() in ["bonjour", "hi", "hello"]:
                            response_message["message"]["text"] = "👋 Bonjour ! Comment puis-je vous aider ?"
                        elif user_text.lower() == "alors?":
                            response_message["message"]["text"] = "Je suis prêt ! Envoyez-moi une image pour une prédiction."
                        else:
                            response_message["message"]["text"] = "Je ne comprends pas encore. Essayez 'bonjour' ou envoyez une image !"

                    elif "attachments" in event["message"]:
                        # Gérer les images
                        for attachment in event["message"]["attachments"]:
                            if attachment["type"] == "image":
                                image_url = attachment["payload"]["url"]
                                print(f"Téléchargement de l'image depuis : {image_url}")
                                response = requests.get(image_url)
                                if response.status_code != 200:
                                    print(f"Erreur téléchargement image : {response.status_code} - {response.text}")
                                    raise HTTPException(status_code=500, detail="Erreur téléchargement image")
                                image = Image.open(BytesIO(response.content)).convert("RGB")
                                print("Image téléchargée et convertie avec succès")
                                input_tensor = preprocess_image(image)
                                input_name = model.get_inputs()[0].name
                                outputs = model.run(None, {input_name: input_tensor})
                                predictions = outputs[0]
                                print(f"Forme des prédictions : {predictions.shape}")  # Log pour déboguer

                                # Vérifier la forme des prédictions
                                if predictions.shape != (1, 1000):
                                    print(f"Erreur : Prédictions inattendues, forme {predictions.shape}, attendu (1, 1000)")
                                    response_message["message"]["text"] = "Erreur lors de la prédiction de l'image. Le modèle ne fonctionne pas comme prévu."
                                else:
                                    top_index = int(np.argmax(predictions))
                                    confidence = float(predictions[0, top_index])
                                    response_message["message"]["text"] = f"Prediction: Classe #{top_index}, Confiance: {confidence:.2%}"

                # Envoyer la réponse à Messenger
                if response_message["message"]["text"]:
                    print(f"Envoi de la réponse à Messenger : {response_message}")
                    fb_response = requests.post(
                        "https://graph.facebook.com/v17.0/me/messages",
                        params={"access_token": PAGE_ACCESS_TOKEN},
                        json=response_message
                    )
                    if fb_response.status_code != 200:
                        print(f"Erreur envoi message Messenger : {fb_response.status_code} - {fb_response.text}")
                        raise HTTPException(status_code=500, detail="Erreur envoi message")
                    print("Message envoyé avec succès à Messenger")

        return {"status": "ok"}
    except Exception as e:
        print(f"Erreur dans le webhook : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur dans le webhook : {e}")
