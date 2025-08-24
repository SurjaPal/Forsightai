from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from main import fire_detector
import requests

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/tweet-officials",methods=["POST"])
def tweet_officials():
    msg = request.json.get("message")
    # Here you would add the logic to send a tweet using Twitter's API
    BOT_TOKEN = "8255507831:AAFO4mmUepxbPTUWpJDe4I77kXIoXq29uNU"
    CHAT_ID = "-4897536546"
    MESSAGE = msg or "Fire detected! Immediate action required."
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    payload = {
        "chat_id": CHAT_ID,
        "text": MESSAGE
    }
    
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print("Failed to send message:", response.text)
    

@app.route('/detect-fire', methods=['POST'])
def detect_fire():
    if fire_detector is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        result = fire_detector.predict(file_path)
        response = {
            "fireDetected": result['is_fire'],
            "confidence": round(result['confidence'] * 100, 2),
            "rawScore": round(result['raw_score'], 4)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
