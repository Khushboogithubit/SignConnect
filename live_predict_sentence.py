import cv2
import mediapipe as mp
import pickle
import time
from grammar_corrector import polish_sentence  # Your grammar correction module
import numpy as np

# ----------------- LOAD MODEL -----------------
with open("model/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------- MEDIAPIPE SETUP -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ----------------- WEBCAM SETUP -----------------
cap = cv2.VideoCapture(0)

# ----------------- SENTENCE STORAGE -----------------
sentence = []
last_word = ""
last_word_time = time.time()
last_correction_time = time.time()
polished = ""

# Update sentence only if word changes
def update_sentence(word):
    global last_word, last_word_time, sentence
    if word != last_word and time.time() - last_word_time > 1:
        sentence.append(word)
        last_word = word
        last_word_time = time.time()

print("🎥 Live Sign Prediction Running — Press 'q' to quit.")

# ----------------- MAIN LOOP -----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    row = []
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            row.extend([v for p in lm.landmark for v in (p.x, p.y, p.z)])

        # Pad to 126 features if only one hand
        if len(row) == 63:
            row.extend([0.0]*63)

    # Predict only if row has 126 features
    if len(row) == 126:
        pred = model.predict([row])[0]
        update_sentence(pred)
        cv2.putText(frame, f"{pred}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Grammar correction every 2 seconds
    raw_text = " ".join(sentence[-5:])  # last 5 words
    if time.time() - last_correction_time > 2:
        polished = polish_sentence(raw_text)
        last_correction_time = time.time()

    # Display corrected sentence
    cv2.putText(frame, polished, (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Live Sign to Sentence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
