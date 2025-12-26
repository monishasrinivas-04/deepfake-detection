import numpy as np
from plausibility_score import compute_plausibility
from normalize_scores import min_max_norm
from fusion_model import fuse_scores
from decision import classify
from evaluation import evaluate

# Load inputs
cnn_scores = np.load("cnn_scores.npy")
geometry_scores = np.load("geometry_scores.npy")
labels = np.load("labels.npy")

# Normalize
cnn_scores = min_max_norm(cnn_scores)
geometry_scores = min_max_norm(geometry_scores)

# Biological plausibility
bio_scores = compute_plausibility(geometry_scores)

# Fusion
fused_scores = fuse_scores(cnn_scores, bio_scores)

# Decision
predictions = classify(fused_scores)

# Evaluation
evaluate(labels, fused_scores)

print("System run completed successfully.")
