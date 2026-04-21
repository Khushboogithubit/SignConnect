from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, mediapipe as mp, pickle, numpy as np, base64, threading, time
from grammar_corrector import polish_sentence as correct_sentence
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# ---------------- MODEL LOAD ----------------
with open("model/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model Loaded")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- SENTENCE ----------------
sentence = []
last_word = ""
last_word_time = time.time()
polished = ""
lock = threading.Lock()

def update_sentence(word):
    global last_word, last_word_time, sentence
    if word != last_word and time.time() - last_word_time > 1:
        sentence.append(word)
        last_word = word
        last_word_time = time.time()

# ---------------- BACKGROUND GRAMMAR ----------------
def background_corrector():
    global polished
    while True:
        if len(sentence) >= 2:
            raw_text = " ".join(sentence[-5:])
            try:
                new_text = correct_sentence(raw_text)
                with lock:
                    polished = new_text
            except:
                pass
        time.sleep(3)

threading.Thread(target=background_corrector, daemon=True).start()

# ---------------- PREDICTION ----------------
def predict_from_frame(base64_image):
    image_data = base64.b64decode(base64_image.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "..."

    if result.multi_hand_landmarks:
        print("✋ Hand detected")
        row = []

        for lm in result.multi_hand_landmarks:
            row.extend([v for p in lm.landmark for v in (p.x, p.y, p.z)])

        if len(row) == 63:
            row.extend([0.0]*63)

        if len(row) == 126:
            try:
                prob = model.predict_proba([row])
                confidence = max(prob[0])

                print(f"Confidence: {confidence}")

                if confidence > 0.6:
                    prediction = model.predict([row])[0]
                    print(f"Prediction: {prediction}")
                    update_sentence(prediction)
            except Exception as e:
                print("Prediction error:", e)

    else:
        print("❌ No hand detected")

    with lock:
        corrected = polished

    return prediction, corrected

@app.route('/')
def home():
    return "Server is working ✅"
# ---------------- ROUTES ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64_image = data.get("image")
    print("🔥 API CALLED")

    if not base64_image:
        return jsonify({"error": "No image"}), 400

    pred, corrected = predict_from_frame(base64_image)

    return jsonify({
        "prediction": pred,
        "corrected": corrected
    })


@app.route('/reset', methods=['POST'])
def reset():
    global sentence
    sentence = []
    return jsonify({"message": "Sentence cleared"})

# ---------------- RUN ----------------
if __name__ == '__main__':
    print("🚀 Flask Server Starting...")
    app.run(port=5000, debug=False)