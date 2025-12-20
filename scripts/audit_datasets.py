import os
import pandas as pd
from collections import Counter

DATA_ROOT = "data/raw"

records = []

for root, _, files in os.walk(DATA_ROOT):
    for f in files:
        if f.endswith((".mp4", ".avi")):
            path = os.path.join(root, f)
            label = "fake" if "fake" in root.lower() else "real"
            source = "DFDC" if "dfdc" in root.lower() else "FaceForensics++"
            records.append([f, label, source, path])

df = pd.DataFrame(records, columns=["video", "label", "source", "path"])

print("Total videos:", len(df))
print("Class distribution:", Counter(df["label"]))
print("Source distribution:", Counter(df["source"]))

os.makedirs("results", exist_ok=True)
df.to_csv("results/metadata.csv", index=False)

print("Saved metadata to results/metadata.csv")
