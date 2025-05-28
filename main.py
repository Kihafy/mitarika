from flask import Flask, request
import requests
import os

app = Flask(__name__)

# 🔑 Tokens : assure-toi qu'ils sont bien définis dans ton environnement ou en dur pour tester
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "mon_token_secret")  # à remplacer si besoin
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "ton_token_page")  # à remplacer si besoin

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
    print("📩 Requête reçue de Facebook :", data)  # pour vérifier la communication

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender_id = messaging_event["sender"]["id"]
            if "message" in messaging_event:
                message_text = messaging_event["message"].get("text")
                print(f"📨 Message reçu de {sender_id} : {message_text}")
                send_message(sender_id, f"Tu as dit : {message_text}")
    return "ok", 200

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
