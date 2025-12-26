import os
import cv2
import numpy as np
from tqdm import tqdm
from mediapipe.python.solutions.face_mesh import FaceMesh

# ----------------------------
# PROJECT ROOT (ROBUST)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


# ---------------------------
# PATHS
# ---------------------------
REAL_FRAMES = os.path.join(
    PROJECT_ROOT, "results", "final_demo", "input", "frames", "real"
)

FAKE_FRAMES = os.path.join(
    PROJECT_ROOT, "results", "final_demo", "input", "frames", "fake"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT, "results", "final_demo", "geometry"
)

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(REAL_FRAMES):
    raise FileNotFoundError(f"REAL_FRAMES not found: {REAL_FRAMES}")

if not os.path.exists(FAKE_FRAMES):
    raise FileNotFoundError(f"FAKE_FRAMES not found: {FAKE_FRAMES}")


# ----------------------------
# MEDIAPIPE INIT
# ----------------------------
face_mesh = FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3
)

# ----------------------------
# UTILS
# ----------------------------
def dist(a, b):
    return np.linalg.norm(a - b)

def extract_geometry(img):
    img = cv2.resize(img, (640, 640))
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h, p.z] for p in lm])

    left_cheek = pts[234]
    right_cheek = pts[454]
    jaw_left = pts[172]
    jaw_right = pts[397]
    nose_left = pts[94]
    nose_right = pts[331]

    face_width = dist(left_cheek, right_cheek)
    if face_width <= 0:
        return None

    jaw_width = dist(jaw_left, jaw_right)
    nose_width = dist(nose_left, nose_right)

    return np.array([
        jaw_width / face_width,
        nose_width / face_width
    ])

# ----------------------------
# PROCESS FOLDER
# ----------------------------
def process(folder, label):
    vectors = []

    for f in tqdm(sorted(os.listdir(folder)), desc=f"Processing {label}"):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)

        if img is None:
            continue

        geom = extract_geometry(img)
        if geom is not None:
            vectors.append(geom)

    vectors = np.array(vectors)
    np.save(os.path.join(OUT_DIR, f"{label}_geometry.npy"), vectors)
    print(f"{label}: saved {vectors.shape[0]} samples")

# ----------------------------
# RUN
# ----------------------------
process(REAL_FRAMES, "real")
process(FAKE_FRAMES, "fake")

print("Geometry extraction completed successfully.")
