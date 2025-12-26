import numpy as np

def compute_plausibility(geometry_scores):
    """
    Converts geometry inconsistency → plausibility score
    """
    # Higher geometry score = more inconsistent
    plausibility = 1.0 - geometry_scores

    # Clip for safety
    plausibility = np.clip(plausibility, 0, 1)
    return plausibility
