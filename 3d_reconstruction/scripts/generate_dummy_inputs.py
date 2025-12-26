import numpy as np

np.random.seed(42)
N = 200

# CNN deepfake probability (0–1)
cnn_scores = np.random.beta(2, 2, N)

# Geometry inconsistency score (0–1, higher = more fake)
geometry_scores = np.random.beta(2, 5, N)

# Ground truth labels
labels = np.array([0] * (N // 2) + [1] * (N // 2))

np.save("cnn_scores.npy", cnn_scores)
np.save("geometry_scores.npy", geometry_scores)
np.save("labels.npy", labels)

print("Dummy inputs created.")
