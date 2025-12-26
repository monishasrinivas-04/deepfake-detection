import numpy as np

# Example landmark indices (DECA standard mesh)
LANDMARKS = {
    "jaw_left": 234,     # approximate
    "jaw_right": 454,
    "nose_left": 327,
    "nose_right": 97,
    "chin": 152,
    "forehead": 10
}

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def extract_measurements(vertices):
    """
    vertices: (N, 3) numpy array from DECA output
    """
    jaw_width = euclidean(
        vertices[LANDMARKS["jaw_left"]],
        vertices[LANDMARKS["jaw_right"]]
    )

    nose_width = euclidean(
        vertices[LANDMARKS["nose_left"]],
        vertices[LANDMARKS["nose_right"]]
    )

    face_height = euclidean(
        vertices[LANDMARKS["chin"]],
        vertices[LANDMARKS["forehead"]]
    )

    return {
        "jaw_width": jaw_width,
        "nose_width": nose_width,
        "face_height": face_height
    }
