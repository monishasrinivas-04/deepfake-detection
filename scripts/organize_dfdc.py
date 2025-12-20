import os
import json
import shutil

DFDC_ROOT = "data/raw/DFDC"
META_PATH = os.path.join(DFDC_ROOT, "metadata.json")

REAL_DIR = os.path.join(DFDC_ROOT, "real")
FAKE_DIR = os.path.join(DFDC_ROOT, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

for video, info in metadata.items():
    label = info.get("label", "").upper()
    src_path = os.path.join(DFDC_ROOT, video)

    if not os.path.exists(src_path):
        continue

    if label == "REAL":
        shutil.move(src_path, os.path.join(REAL_DIR, video))
    elif label == "FAKE":
        shutil.move(src_path, os.path.join(FAKE_DIR, video))

print("DFDC organization completed.")
