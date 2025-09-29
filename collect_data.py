import cv2
import mediapipe as mp
import csv, os

# ----------------- SETUP -----------------
label = input("Enter label: ").strip().lower()
os.makedirs("data", exist_ok=True)
csv_path = f"data/{label}.csv"
csv_file = open(csv_path, "a", newline="")
writer = csv.writer(csv_file)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ----------------- CHOOSE MODE -----------------
print("\nChoose input mode:")
print("1. Images from 'uploads/images'")
print("2. Videos from 'uploads/videos'")
print("3. Real-time Camera")
choice = input("Enter (1/2/3): ").strip()

# ----------------- PROCESS IMAGE -----------------
def process_image(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            row = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
            writer.writerow(row)

# ----------------- PROCESS VIDEO -----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                row = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
                writer.writerow(row)
    cap.release()

# ----------------- PROCESS CAMERA -----------------
def process_camera():
    cap = cv2.VideoCapture(0)  # webcam
    print("Press 'q' to quit camera...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                row = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
                writer.writerow(row)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# ----------------- MAIN LOGIC -----------------
if choice == "1":  # Images
    folder = "uploads/images"
    os.makedirs(folder, exist_ok=True)
    files = [f"{folder}/{f}" for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in files:
        process_image(img)
    print(f"✅ Processed {len(files)} images.")

elif choice == "2":  # Videos
    folder = "uploads/videos"
    os.makedirs(folder, exist_ok=True)
    files = [f"{folder}/{f}" for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    for vid in files:
        process_video(vid)
        print(f"🎥 Done: {vid}")
    print(f"✅ Processed {len(files)} videos.")

elif choice == "3":  # Camera
    process_camera()
    print("✅ Camera session completed.")

else:
    print("❌ Invalid choice.")

csv_file.close()
print(f"\n📁 Landmark data saved to {csv_path}")
