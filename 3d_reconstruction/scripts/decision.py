import numpy as np

def classify(scores, threshold=0.5):
    return (scores >= threshold).astype(int)
