import numpy as np
from dna_priors import build_priors
from deca_measurements import extract_measurements
from prior_validation import biological_plausibility_score

# Load DECA output
vertices = np.load("deca_vertices.npy")  # shape (N, 3)

# Step 1: Build priors
priors = build_priors()

# Step 2: Extract geometry traits
measurements = extract_measurements(vertices)

# Step 3: Validate
score, details = biological_plausibility_score(measurements, priors)

print("Biological Plausibility Score:", score)
for trait, info in details.items():
    print(trait, info)
