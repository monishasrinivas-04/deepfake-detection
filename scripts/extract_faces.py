import os
import cv2

DATA_ROOT = "data/raw"
OUT_ROOT = "data/processed/faces"
FRAME_INTERVAL = 10
IMG_SIZE = 224

# Load Haar Cascade
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            

        if frame_id % FRAME_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                out_dir = os.path.join(OUT_ROOT, label)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, f"{saved}.jpg"), face)
                saved += 1
                

        frame_id += 1

    cap.release()

def main():
    count = 0
    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".mp4"):
                label = "fake" if "fake" in root.lower() else "real"
                extract_faces(os.path.join(root, f), label)
                count += 1
                if count == 50:
                    return

    
if __name__ == "__main__":
    main()

def extract_faces(video_path, label):
    print(f"Processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
