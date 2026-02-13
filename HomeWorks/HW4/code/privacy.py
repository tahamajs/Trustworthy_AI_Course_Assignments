"""Privacy (Q2) — Laplace mechanism helpers and assignment scenario calculators."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


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


def average_income_scale(sensitivity: float, epsilon: float, n: int) -> float:
    """Scale for average-income query with provided sensitivity."""
    return laplace_scale(sensitivity, epsilon)


def income_query_results(
    epsilon: float = 0.1,
    sensitivity_avg: float = 5000.0,
    sensitivity_total: float = 50000.0,
    true_avg: float = 40000.0,
    true_total: float = 20_000_000.0,
    sampled_noise_avg: float = 2000.0,
    sampled_noise_total: float = 5000.0,
    epsilon_avg_split: float = 0.05,
    epsilon_total_split: float = 0.05,
) -> Dict[str, float]:
    """Deterministic outputs for HW4 Q2-Part1.

    For split-budget results, we rescale the provided noise sample by the ratio of scales so
    the effect of changed epsilon is explicit and reproducible.
    """
    b_avg = laplace_scale(sensitivity_avg, epsilon)
    b_total = laplace_scale(sensitivity_total, epsilon)

    noisy_avg = true_avg + sampled_noise_avg
    noisy_total = true_total + sampled_noise_total

    b_avg_split = laplace_scale(sensitivity_avg, epsilon_avg_split)
    b_total_split = laplace_scale(sensitivity_total, epsilon_total_split)

    z_avg = sampled_noise_avg / b_avg
    z_total = sampled_noise_total / b_total
    sampled_noise_avg_split = z_avg * b_avg_split
    sampled_noise_total_split = z_total * b_total_split

    noisy_avg_split = true_avg + sampled_noise_avg_split
    noisy_total_split = true_total + sampled_noise_total_split

    return {
        "epsilon_total_basic_composition": compose_epsilons([epsilon_avg_split, epsilon_total_split]),
        "b_avg": b_avg,
        "b_total": b_total,
        "noisy_avg": noisy_avg,
        "noisy_total": noisy_total,
        "b_avg_split": b_avg_split,
        "b_total_split": b_total_split,
        "sampled_noise_avg_split": sampled_noise_avg_split,
        "sampled_noise_total_split": sampled_noise_total_split,
        "noisy_avg_split": noisy_avg_split,
        "noisy_total_split": noisy_total_split,
    }


def counting_query_results(
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    true_value: float = 500.0,
    threshold: float = 505.0,
    k: int = 92,
    n: int = 500,
    p: float = 0.01,
) -> Dict[str, float]:
    """Deterministic outputs for HW4 Q2-Part2.

    Sequential assumption is locked to epsilon_i = epsilon / k and delta_i = delta / k.
    Unbounded sensitivity is locked to max(1, ceil(p*n)) * sensitivity.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    if p < 0:
        raise ValueError("p must be >= 0")

    b_base = laplace_scale(sensitivity, epsilon)
    prob_base = laplace_cdf_threshold(threshold, true_value, sensitivity, epsilon)

    epsilon_i = epsilon / k
    delta_i = delta / k
    b_sequential = laplace_scale(sensitivity, epsilon_i)
    prob_sequential = laplace_cdf_threshold(threshold, true_value, sensitivity, epsilon_i)

    delta_f_unbounded = max(1, int(math.ceil(p * n))) * sensitivity
    b_unbounded = laplace_scale(delta_f_unbounded, epsilon_i)
    prob_unbounded = laplace_cdf_threshold(threshold, true_value, delta_f_unbounded, epsilon_i)

    return {
        "epsilon": float(epsilon),
        "delta": float(delta),
        "k": int(k),
        "epsilon_i": float(epsilon_i),
        "delta_i": float(delta_i),
        "b_base": float(b_base),
        "prob_base_gt_threshold": float(prob_base),
        "b_sequential": float(b_sequential),
        "prob_sequential_gt_threshold": float(prob_sequential),
        "delta_f_unbounded": float(delta_f_unbounded),
        "b_unbounded": float(b_unbounded),
        "prob_unbounded_gt_threshold": float(prob_unbounded),
        "n": int(n),
        "p": float(p),
    }


if __name__ == "__main__":
    inc = income_query_results()
    cnt = counting_query_results()
    print("Q2 Part1:", inc)
    print("Q2 Part2:", cnt)
