def check_trait(value, prior_range):
    return prior_range[0] <= value <= prior_range[1]

def biological_plausibility_score(measurements, priors):
    results = {}
    valid_count = 0

    for trait, value in measurements.items():
        low, high = priors[trait]["range_95"]
        is_valid = check_trait(value, (low, high))
        results[trait] = {
            "value_mm": value,
            "valid_range": (low, high),
            "plausible": is_valid
        }
        if is_valid:
            valid_count += 1

    score = valid_count / len(measurements)
    return score, results
