"""Privacy (Q2) — Laplace mechanism helpers and example calculations."""
import math
from typing import Tuple
import numpy as np
from scipy.stats import laplace


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return float(sensitivity / epsilon)


def add_laplace_noise(value: float, sensitivity: float, epsilon: float, rng=None) -> float:
    b = laplace_scale(sensitivity, epsilon)
    rng = np.random.default_rng() if rng is None else rng
    noise = rng.laplace(0.0, b)
    return float(value + noise)


def laplace_cdf_threshold(threshold: float, true_value: float, sensitivity: float, epsilon: float) -> float:
    """Probability that noisy response > threshold using Laplace(false_value, b).

    P( true_value + Lap(b) > threshold ) = 1 - CDF_Laplace(threshold - true_value; 0, b)
    """
    b = laplace_scale(sensitivity, epsilon)
    # CDF of Laplace(0, b) at x is 0.5*(1 + sign(x)*(1 - exp(-|x|/b)))
    x = threshold - true_value
    if x >= 0:
        cdf = 1 - 0.5 * math.exp(-x / b)
    else:
        cdf = 0.5 * math.exp(x / b)
    return float(1 - cdf)


def compose_epsilons(epsilons: list) -> float:
    """Simple (basic) composition: eps_total = sum(epsilons)"""
    return float(sum(epsilons))


# --- Example helper functions used in the assignment text ---
def average_income_scale(sensitivity: float, epsilon: float, n: int) -> float:
    """Example: scale for average income when sensitivity is provided (keeps interface consistent)."""
    return laplace_scale(sensitivity, epsilon)


if __name__ == "__main__":
    # Quick deterministic example
    sens_avg = 200000 / 500.0  # example sensitivity for average (placeholder)
    eps = 1.0
    print("Laplace scale for example avg:", laplace_scale(sens_avg, eps))
