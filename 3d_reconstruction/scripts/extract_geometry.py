import trimesh
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(BASE_DIR, "outputs", "meshes")
OUT_DIR = "../outputs/geometry_vectors"
os.makedirs(OUT_DIR, exist_ok=True)

# Example landmark indices (document in report)
LEFT_CHEEK = 2345
RIGHT_CHEEK = 4543
JAW_LEFT = 1500
JAW_RIGHT = 3000
NOSE_LEFT = 3300
NOSE_RIGHT = 3400

def dist(a, b):
    return np.linalg.norm(a - b)

for mesh_file in os.listdir(MESH_DIR):
    mesh = trimesh.load(os.path.join(MESH_DIR, mesh_file))
    v = mesh.vertices

    face_width = dist(v[LEFT_CHEEK], v[RIGHT_CHEEK])
    jaw_width = dist(v[JAW_LEFT], v[JAW_RIGHT])
    nose_width = dist(v[NOSE_LEFT], v[NOSE_RIGHT])

    geom = np.array([
        face_width,
        jaw_width,
        nose_width,
        jaw_width / face_width,
        nose_width / face_width
    ])

    np.save(
        os.path.join(OUT_DIR, mesh_file.replace(".obj", ".npy")),
        geom
    )

print("Geometry extraction completed.")
