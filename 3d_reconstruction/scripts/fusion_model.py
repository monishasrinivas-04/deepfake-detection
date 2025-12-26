import numpy as np

def fuse_scores(cnn, bio, w_cnn=0.6, w_bio=0.4):
    """
    Weighted score fusion
    """
    fused = w_cnn * cnn + w_bio * (1 - bio)
    return np.clip(fused, 0, 1)
