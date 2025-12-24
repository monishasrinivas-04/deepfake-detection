import os
import sys
import cv2
import torch
import numpy as np


# -----------------------
# SAFETY CHECKS
# -----------------------
assert torch.cuda.is_available() is False, "GPU should be disabled for safety"

# -----------------------
# PATHS
# -----------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DECA_PATH = os.path.join(ROOT, "DECA")

if not os.path.exists(DECA_PATH):
    raise FileNotFoundError("DECA folder not found at repo root")

sys.path.append(DECA_PATH)

from decalib.deca import DECA
from decalib.utils.config import cfg

# Disable renderer (avoids pytorch3d on Windows)
cfg.rasterizer_type = "none"

INPUT_DIR = os.path.join(ROOT, "3d_reconstruction", "data", "sampled_frames")
OUT_MESH = os.path.join(ROOT, "3d_reconstruction", "outputs", "meshes")
OUT_SHAPE = os.path.join(ROOT, "3d_reconstruction", "outputs", "shape_vectors")

os.makedirs(OUT_MESH, exist_ok=True)
os.makedirs(OUT_SHAPE, exist_ok=True)

# -----------------------
# INIT DECA (CPU ONLY)
# -----------------------
cfg.model.use_tex = False
device = torch.device("cpu")
# Initialize DECA WITHOUT renderer (Windows-safe)
cfg.rasterizer_type = None

from decalib.deca import DECA as DECA_CLASS

class DECA_NoRender(DECA_CLASS):
    def _setup_renderer(self, model_cfg):
        # Skip renderer setup entirely
        self.render = None

deca = DECA_NoRender(config=cfg, device=device)

# -----------------------
# PROCESS FRAMES
# -----------------------

for video in os.listdir(INPUT_DIR):
    video_dir = os.path.join(INPUT_DIR, video)
    if not os.path.isdir(video_dir):
        continue

    for img_name in os.listdir(video_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(video_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        img_tensor = torch.from_numpy(img / 255.0).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            codedict = deca.encode(img_tensor)
            opdict, _ = deca.decode(codedict)

        # Save shape coefficients
        shape = codedict["shape"].cpu().numpy()
        np.save(
            os.path.join(OUT_SHAPE, f"{video}_{img_name}.npy"),
            shape
        )

        # Save mesh
        verts = opdict["verts"][0].cpu().numpy()
        faces = deca.flame.faces_tensor.cpu().numpy()

        mesh_file = os.path.join(OUT_MESH, f"{video}_{img_name}.obj")
        with open(mesh_file, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces + 1:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

print("DECA reconstruction completed successfully.")
