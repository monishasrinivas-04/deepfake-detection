import numpy as np

# =========================
# 1. Simulated DNA Priors
# =========================

TRAIT_PRIORS = {
    "jaw_width": {"mean": 120, "std": 6, "h2": 0.40},
    "nose_width": {"mean": 36, "std": 3, "h2": 0.30},
    "face_height": {"mean": 185, "std": 8, "h2": 0.45}
}

def build_priors(n=10000):
    priors = {}
    for trait, p in TRAIT_PRIORS.items():
        G = np.random.normal(0, 1, n)
        E = np.random.normal(0, 1, n)

        values = p["mean"] + p["std"] * (
            np.sqrt(p["h2"]) * G +
            np.sqrt(1 - p["h2"]) * E
        )

        priors[trait] = {
            "mean": np.mean(values),
            "var": np.var(values),
            "range_95": np.percentile(values, [2.5, 97.5])
        }
    return priors


# =========================
# 2. Fake DECA Vertices
# =========================
# DECA mesh ~5023 vertices → simulate geometry in mm

np.random.seed(42)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vertices_path",
    type=str,
    required=True,
    help="Path to .npy vertex array"
)
args = parser.parse_args()

vertices = np.load(args.vertices_path)

# landmark indices (DECA-consistent positions)
LM = {
    "jaw_left": 234,
    "jaw_right": 454,
    "nose_left": 327,
    "nose_right": 97,
    "chin": 152,
    "forehead": 10
}

def dist(a, b):
    return np.linalg.norm(a - b)

def extract_measurements(verts):
    return {
        "jaw_width": dist(verts[LM["jaw_left"]], verts[LM["jaw_right"]]),
        "nose_width": dist(verts[LM["nose_left"]], verts[LM["nose_right"]]),
        "face_height": dist(verts[LM["chin"]], verts[LM["forehead"]])
    }


# =========================
# 3. Prior Validation
# =========================

def validate(measurements, priors):
    results = {}
    score = 0

    for trait, value in measurements.items():
        low, high = priors[trait]["range_95"]
        plausible = low <= value <= high
        score += int(plausible)

        results[trait] = {
            "measured_mm": round(value, 2),
            "valid_range_mm": (round(low, 2), round(high, 2)),
            "plausible": plausible
        }

    return score / len(measurements), results


# =========================
# 4. RUN PIPELINE
# =========================

if __name__ == "__main__":
    priors = build_priors()
    measurements = extract_measurements(vertices)
    score, report = validate(measurements, priors)

    print("\n=== BIOLOGICAL PLAUSIBILITY REPORT ===\n")
    print("Overall Plausibility Score:", round(score, 2), "\n")

    for trait, info in report.items():
        print(f"{trait.upper()}")
        print("  Measured:", info["measured_mm"], "mm")
        print("  Valid range:", info["valid_range_mm"], "mm")
        print("  Plausible:", info["plausible"])
        print()
