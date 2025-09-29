from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import re
import pickle
from grammar_corrector import correct_sentence  # ✅ Ensure this file exists and works

app = Flask(__name__)
CORS(app)  # ✅ Allow requests from any origin (or specify origins=["http://127.0.0.1:5502"] for stricter control)

# Load the trained model
with open("model/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# ✅ Serve signlive.html from 'pages/' folder
@app.route("/")
def serve_signlive():
    return send_from_directory("pages", "signlive.html")

# ✅ Predict gesture from base64 image
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"prediction": "No image", "corrected": "No image"})

    # Decode base64 image
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    try:
        image = base64.b64decode(image_data)
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"prediction": "Error decoding image", "corrected": str(e)})

    # Mirror and convert image
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image using MediaPipe
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            row = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
            if len(row) == 63:
                try:
                    pred = model.predict([row])[0]
                    corrected = correct_sentence(pred)
                    return jsonify({"prediction": pred, "corrected": corrected})
                except Exception as e:
                    return jsonify({"prediction": "Prediction error", "corrected": str(e)})

    return jsonify({"prediction": "No Hand", "corrected": "No Hand"})

# ✅ Run on port 5000 for API
if __name__ == "__main__":
    app.run(debug=True, port=5000)
