from flask import Flask, request
import requests
import os
from transformers import pipeline

app = Flask(__name__)

# 🔑 Tokens : assure-toi qu'ils sont bien définis dans ton environnement ou en dur pour tester
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "mon_token_secret")
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "ton_token_page")

# Initialiser le pipeline Hugging Face pour la conversation
generator = pipeline("text-generation", model="facebook/blenderbot-400M-distill")

# Contexte pour spécialiser le bot sur les plantes de potager
CONTEXT = (
    "Tu es un expert en jardinage, spécialisé dans les plantes de potager comme les tomates, les courgettes, les carottes, les salades, etc. "
    "Tu donnes des conseils pratiques et réponds de manière amicale et naturelle, comme un ami jardinier qui partage son savoir."
)

@app.route("/", methods=["GET"])
def verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
        print("✅ Webhook vérifié avec succès.")
        return request.args.get("hub.challenge")
    print("❌ Échec de la vérification du webhook.")
    return "Erreur de vérification"

@app.route("/", methods=["POST"])
def webhook():
    data = request.get_json()
    print("📩 Requête reçue de Facebook :", data)

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender_id = messaging_event["sender"]["id"]
            if "message" in messaging_event:
                message_text = messaging_event["message"].get("text")
                if message_text:  # Vérifie que le message n'est pas vide
                    print(f"📨 Message reçu de {sender_id} : {message_text}")
                    # Générer une réponse avec le modèle Hugging Face
                    response_text = generate_response(message_text)
                    send_message(sender_id, response_text)
    return "ok", 200

def generate_response(input_text):
    try:
        # Ajouter le contexte au texte d'entrée pour orienter la réponse
        prompt = f"{CONTEXT} Utilisateur : {input_text}"
        # Générer une réponse avec le modèle
        generated = generator(
            prompt,
            max_length=100,  # Longueur maximale pour des réponses concises
            num_return_sequences=1,  # Une seule réponse
            truncation=True,  # Éviter les erreurs de longueur
            pad_token_id=generator.tokenizer.eos_token_id  # Token de fin pour BlenderBot
        )
        response_text = generated[0]["generated_text"].strip()
        # Supprimer le contexte et l'entrée de l'utilisateur de la réponse
        if response_text.startswith(CONTEXT):
            response_text = response_text[len(CONTEXT):].strip()
        if response_text.startswith(f"Utilisateur : {input_text}"):
            response_text = response_text[len(f"Utilisateur : {input_text}"):].strip()
        # Limiter la réponse à une phrase ou deux pour plus de naturel
        response_text = response_text.split(". ")[0] + "." if ". " in response_text else response_text
        return response_text if response_text else "Désolé, je n'ai pas bien compris. Peux-tu préciser ?"
    except Exception as e:
        print(f"❌ Erreur lors de la génération de la réponse : {e}")
        return "Oups, quelque chose s'est mal passé ! Peux-tu réessayer ?"

def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v17.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }

    response = requests.post(url, headers=headers, json=payload)
    print(f"📤 Message envoyé à {recipient_id} : {message_text}")
    print("📥 Réponse de l'API Facebook :", response.status_code, response.text)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
