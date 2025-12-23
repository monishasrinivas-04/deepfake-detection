import os
import shutil
import csv

DFDC_ROOT = "data/raw/DFDC"
FFPP_ROOT = "data/raw/FaceForensics++"  # optional
OUT_ROOT = "data/merged"

REAL_DIR = os.path.join(OUT_ROOT, "real")
FAKE_DIR = os.path.join(OUT_ROOT, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

metadata = []

def copy_videos(src_root, dataset_name):
    for label in ["real", "fake"]:
        src_dir = os.path.join(src_root, label)
        if not os.path.exists(src_dir):
            continue

        for f in os.listdir(src_dir):
            if not f.endswith(".mp4"):
                continue

            new_name = f"{dataset_name}_{f}"
            dst_dir = REAL_DIR if label == "real" else FAKE_DIR

            shutil.copy(
                os.path.join(src_dir, f),
                os.path.join(dst_dir, new_name)
            )

            metadata.append([new_name, label, dataset_name])

# Merge DFDC
copy_videos(DFDC_ROOT, "DFDC")

# Merge FaceForensics++ if available
if os.path.exists(FFPP_ROOT):
    copy_videos(FFPP_ROOT, "FFPP")

# Save metadata
with open(os.path.join(OUT_ROOT, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "source"])
    writer.writerows(metadata)

print("Dataset merge completed.")
