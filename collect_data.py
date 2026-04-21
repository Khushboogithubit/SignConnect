
import cv2
import mediapipe as mp
import csv, os, time
import numpy as np

# ----------------- SETUP -----------------
label = input("Enter label (word): ").strip().lower()
os.makedirs("data", exist_ok=True)
csv_path = f"data/{label}.csv"
csv_file = open(csv_path, "a", newline="")
writer = csv.writer(csv_file)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Press 'c' to capture a 2s clip, 'q' to quit.")

# ----------------- FUNCTION: RECORD CLIP -----------------
def record_clip(duration=2):
    start = time.time()
    clip_data = []

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        row = []
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                row.extend([v for p in lm.landmark for v in (p.x, p.y, p.z)])

        # Pad to 126 values if only one hand detected
        if len(row) == 63:
            row.extend([0.0]*63)

        if row:  # at least one hand detected
            clip_data.append(row)


        cv2.imshow("Clip Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None  # exit

    if clip_data:
        # Average landmarks over the clip → one row
        clip_array = np.mean(np.array(clip_data), axis=0)
        writer.writerow(clip_array)
        print(f"✅ Saved one sample for '{label}' ({len(clip_data)} frames).")

# ----------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 'c' to record 2s clip, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Collect Clips", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        print("🎬 Recording 2s clip...")
        record_clip()
    elif key == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"📁 Data saved to {csv_path}")

