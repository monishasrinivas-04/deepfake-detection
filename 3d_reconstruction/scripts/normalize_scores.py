import numpy as np

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
