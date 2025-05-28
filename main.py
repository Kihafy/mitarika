from flask import Flask, request, jsonify
import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from skimage import transform, exposure
from sklearn.cluster import KMeans
import requests

app = Flask(__name__)

# Configuration
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "mon_token_secret")
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "ton_token_page")

# Chargement des modèles ONNX (ultra-légers)
plant_model = ort.InferenceSession("plant_disease.onnx")
compression_model = ort.InferenceSession("image_compressor.onnx")

# Dictionnaire des maladies des plantes courantes à Madagascar
PLANT_DISEASES = {
    0: {"name": "En bonne santé", "advice": "Votre plante est en bonne santé. Continuez à l'arroser régulièrement."},
    1: {"name": "Mildiou", "advice": "Traitement recommandé : Mélangez 1 litre d'eau avec 10ml de savon noir et pulvérisez tous les 3 jours."},
    2: {"name": "Oïdium", "advice": "Préparation naturelle : 1 litre d'eau + 1 cuillère à soupe de bicarbonate de soude. Pulvérisez 2 fois par semaine."},
    3: {"name": "Rouille", "advice": "Coupez les feuilles atteintes. Traitez avec une décoction de prêle (100g/litre) tous les 5 jours."},
    4: {"name": "Carence en azote", "advice": "Ajoutez du compost ou du fumier bien décomposé autour de la plante."}
}

@app.route("/", methods=["GET"])
def verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
        print("✅ Webhook vérifié avec succès.")
        return request.args.get("hub.challenge")
    print("❌ Échec de la vérification du webhook.")
    return "Erreur de vérification", 403

@app.route("/", methods=["POST"])
def webhook():
    data = request.get_json()
    print("📩 Requête reçue de Facebook :", data)

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender_id = messaging_event["sender"]["id"]
            if "message" in messaging_event:
                if "text" in messaging_event["message"]:
                    message_text = messaging_event["message"]["text"]
                    response_text = generate_text_response(message_text)
                    send_message(sender_id, response_text)
                elif "attachments" in messaging_event["message"]:
                    for attachment in messaging_event["message"]["attachments"]:
                        if attachment["type"] == "image":
                            image_url = attachment["payload"]["url"]
                            analysis_result = analyze_plant_image(image_url)
                            send_message(sender_id, analysis_result)
    return "ok", 200

def generate_text_response(input_text):
    # Système de règles simples pour les questions fréquentes
    input_text = input_text.lower()
    
    responses = {
        "bonjour": "Bonjour ami cultivateur ! Comment puis-je vous aider avec vos plantes aujourd'hui ?",
        "merci": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions sur vos plantes.",
        "tomate": "Pour les tomates : espacez les plants de 50cm, arrosez au pied sans mouiller les feuilles, et paillez pour conserver l'humidité.",
        "riz": "Le riz à Madagascar : préférez les variétés locales comme le Makalioka, plantez en décembre-janvier, et alternez eau/assèchement pour limiter les moustiques.",
        "manioc": "Le manioc pousse bien en sol pauvre. Plantez des boutures de 20cm inclinées à 45°. Récolte en 8-12 mois."
    }
    
    for keyword, response in responses.items():
        if keyword in input_text:
            return response
    
    return ("Je suis un expert des plantes malgaches. Envoyez-moi une photo de votre plante pour un diagnostic, "
            "ou posez-moi des questions sur : tomate, riz, manioc, maïs, haricot.")

def analyze_plant_image(image_url):
    try:
        # Téléchargement et compression de l'image
        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        img = preprocess_image(img)
        
        # Analyse avec le modèle ONNX
        input_name = plant_model.get_inputs()[0].name
        output = plant_model.run(None, {input_name: img[np.newaxis, ...]})[0]
        disease_id = np.argmax(output)
        disease = PLANT_DISEASES.get(disease_id, PLANT_DISEASES[0])
        
        # Analyse des couleurs pour des conseils supplémentaires
        color_advice = analyze_colors(img)
        
        return (f"Diagnostic : {disease['name']}\n\n"
                f"Conseil : {disease['advice']}\n\n"
                f"Info supplémentaire : {color_advice}")
    
    except Exception as e:
        print(f"Erreur d'analyse : {e}")
        return "Désolé, je n'ai pas pu analyser l'image. Pouvez-vous envoyer une photo plus claire de la plante ?"

def preprocess_image(img):
    # Compression intelligente adaptée aux smartphones basse performance
    img = img.convert("RGB")
    img = img.resize((224, 224))  # Taille adaptée pour les modèles légers
    
    # Conversion en numpy array et normalisation
    img = np.array(img).astype(np.float32) / 255.0
    
    # Amélioration du contraste pour les conditions lumineuses tropicales
    img = exposure.equalize_hist(img)
    
    return img

def analyze_colors(img):
    # Analyse des couleurs dominantes pour détecter carences
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    
    dominant_colors = kmeans.cluster_centers_
    avg_green = dominant_colors[:, 1].mean()
    avg_yellow = (dominant_colors[:, 0] + dominant_colors[:, 2]).mean() / 2
    
    if avg_green < 0.3:
        return "La plante semble manquer de chlorophylle (feuilles pâles), indiquant possiblement une carence en azote."
    elif avg_yellow > 0.6:
        return "Les teintes jaunes dominantes peuvent indiquer un stress hydrique ou une maladie fongique."
    return "Les couleurs semblent normales pour une plante en bonne santé."

def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    requests.post(url, headers=headers, json=payload)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
