import cv2
import os

VIDEO_DIR = "../../data/videos"   # put some mp4s here
OUT_DIR = "../data/sampled_frames"
FRAME_INTERVAL = 30

os.makedirs(OUT_DIR, exist_ok=True)

for video in os.listdir(VIDEO_DIR):
    if not video.endswith(".mp4"):
        continue

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video))
    frame_id = 0
    saved = 0

    video_name = video.replace(".mp4", "")
    out_subdir = os.path.join(OUT_DIR, video_name)
    os.makedirs(out_subdir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_INTERVAL == 0:
            cv2.imwrite(
                os.path.join(out_subdir, f"frame_{saved}.jpg"),
                frame
            )
            saved += 1

        frame_id += 1

    cap.release()

print("Frame sampling completed.")
