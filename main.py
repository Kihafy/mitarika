from flask import Flask, request
import requests
import os

app = Flask(__name__)

# 🔑 Tokens : assure-toi qu'ils sont définis dans les variables d'environnement sur Render
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "mon_token_secret")
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "ton_token_page")
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "ton_token_huggingface")

# Contexte pour spécialiser le bot sur les plantes de potager et répondre en français
CONTEXT = (
    "Tu es un expert en jardinage, spécialisé dans les plantes de potager comme les tomates, les courgettes, les carottes, les salades, etc. "
    "Réponds en français de manière amicale, naturelle et concise, comme un ami jardinier qui partage son savoir."
)

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
                message_text = messaging_event["message"].get("text")
                if message_text:  # Vérifie que le message n'est pas vide
                    print(f"📨 Message reçu de {sender_id} : {message_text}")
                    # Générer une réponse avec l'API Hugging Face
                    response_text = generate_response(message_text)
                    send_message(sender_id, response_text)
    return "ok", 200

def generate_response(input_text):
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        prompt = f"{CONTEXT} Utilisateur : {input_text}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 100,
                "num_return_sequences": 1,
                "top_p": 0.9,
                "temperature": 0.7  # Pour des réponses naturelles
            }
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Lève une erreur si la requête échoue
        response_text = response.json()[0]["generated_text"].strip()
        # Supprimer le contexte et l'entrée de l'utilisateur
        if response_text.startswith(CONTEXT):
            response_text = response_text[len(CONTEXT):].strip()
        if response_text.startswith(f"Utilisateur : {input_text}"):
            response_text = response_text[len(f"Utilisateur : {input_text}"):].strip()
        # Limiter à une phrase pour plus de concision
        response_text = response_text.split(". ")[0] + "." if ". " in response_text else response_text
        return response_text if response_text else "Désolé, je n'ai pas compris. Peux-tu préciser ?"
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
    port = int(os.environ.get("PORT", 5000))  # Utilise le port fourni par Render
    app.run(host="0.0.0.0", port=port, debug=False)
