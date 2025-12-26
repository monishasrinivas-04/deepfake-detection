import numpy as np

# Literature-informed simulated priors (population-level)
TRAIT_PRIORS = {
    "jaw_width": {
        "mean_mm": 120,
        "std_mm": 6,
        "heritability": 0.40
    },
    "nose_width": {
        "mean_mm": 36,
        "std_mm": 3,
        "heritability": 0.30
    },
    "face_height": {
        "mean_mm": 185,
        "std_mm": 8,
        "heritability": 0.45
    }
}

def generate_prior_distribution(mean, std, heritability, n=10000):
    G = np.random.normal(0, 1, n)      # latent genetic effect
    E = np.random.normal(0, 1, n)      # environment

    values = mean + std * (
        np.sqrt(heritability) * G +
        np.sqrt(1 - heritability) * E
    )
    return values

def build_priors():
    priors = {}
    for trait, params in TRAIT_PRIORS.items():
        samples = generate_prior_distribution(
            params["mean_mm"],
            params["std_mm"],
            params["heritability"]
        )
        priors[trait] = {
            "mean": np.mean(samples),
            "variance": np.var(samples),
            "range_95": np.percentile(samples, [2.5, 97.5])
        }
    return priors

if __name__ == "__main__":
    priors = build_priors()
    for k, v in priors.items():
        print(k, v)
