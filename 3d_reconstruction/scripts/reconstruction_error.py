import os
import argparse
import numpy as np


def load_shape_vectors(shape_dir):
    vectors = []
    for f in os.listdir(shape_dir):
        if f.endswith(".npy"):
            vec = np.load(os.path.join(shape_dir, f))
            vectors.append(vec)
    return np.array(vectors)


def compute_variance(vectors):
    return np.mean(np.var(vectors, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Path to REAL output directory")
    parser.add_argument("--fake", required=True, help="Path to FAKE output directory")
    args = parser.parse_args()

    real_vectors = load_shape_vectors(os.path.join(args.real, "geometry_vectors"))
    fake_vectors = load_shape_vectors(os.path.join(args.fake, "geometry_vectors"))

    real_var = compute_variance(real_vectors)
    fake_var = compute_variance(fake_vectors)

    print("Reconstruction consistency (lower is better):")
    print(f"REAL video variance: {real_var:.6f}")
    print(f"FAKE video variance: {fake_var:.6f}")

    if fake_var > real_var:
        print("→ Fake video shows higher geometric inconsistency")
    else:
        print("→ Geometry consistency is similar")


if __name__ == "__main__":
    main()
