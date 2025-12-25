import os
import cv2
import argparse

def sample_frames(video_path, out_dir, every_n=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % every_n == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_id += 1

    cap.release()
    print(f"Saved {saved} frames from {os.path.basename(video_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output frame directory")
    parser.add_argument("--every_n", type=int, default=10)
    args = parser.parse_args()

    sample_frames(args.video, args.out, args.every_n)
