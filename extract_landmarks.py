import os
import cv2
import pandas as pd
import mediapipe as mp

# Folder containing your videos
video_folder = "videos"  # change this to your video folder
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# Folder to save CSVs
os.makedirs("data", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

for video_file in video_files:
    landmarks = []
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_landmarks = []
                for lm in hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
                landmarks.append(frame_landmarks)

    cap.release()

    # Check if landmarks were found
    if len(landmarks) == 0:
        print(f"No hand landmarks detected in {video_file}. CSV will be empty.")
    else:
        print(f"{video_file}: {len(landmarks)} frames with landmarks.")

    # Save CSV
    save_path = os.path.join("data", video_file.replace(".mp4", ".csv"))
    pd.DataFrame(landmarks).to_csv(save_path, index=False, header=False)
    print("CSV saved at:", os.path.abspath(save_path))

hands.close()
print("📂 All videos processed. Landmark CSVs saved in /data/")

