import numpy as np
import os

SHAPE_DIR = "../outputs/shape_vectors"

video_shapes = {}

for f in os.listdir(SHAPE_DIR):
    video = f.split("_")[0]
    vec = np.load(os.path.join(SHAPE_DIR, f)).squeeze()
    video_shapes.setdefault(video, []).append(vec)

for video, shapes in video_shapes.items():
    shapes = np.array(shapes)
    variance = np.mean(np.var(shapes, axis=0))
    print(f"{video} reconstruction variance: {variance:.4f}")
