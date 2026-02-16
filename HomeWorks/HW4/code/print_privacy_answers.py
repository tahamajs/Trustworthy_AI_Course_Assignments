"""Print report-ready Q2 (Privacy) numerical answers for copy-paste into the report."""

from __future__ import annotations

from privacy import counting_query_results, income_query_results


def main() -> None:
    inc = income_query_results(
        epsilon=0.1,
        sensitivity_avg=5000.0,
        sensitivity_total=50000.0,
        true_avg=40000.0,
        true_total=20_000_000.0,
        sampled_noise_avg=2000.0,
        sampled_noise_total=5000.0,
        epsilon_avg_split=0.05,
        epsilon_total_split=0.05,
    )
    cnt = counting_query_results(
        epsilon=0.1,
        delta=1e-5,
        sensitivity=1.0,
        true_value=500.0,
        threshold=505.0,
        k=92,
        n=500,
        p=0.01,
    )

    print("=" * 60)
    print("Q2 Part 1 — Income queries (Laplace mechanism)")
    print("=" * 60)
    print(f"  b (average income):     {inc['b_avg']:.4f}")
    print(f"  b (total income):       {inc['b_total']:.4f}")
    print(f"  Privacy-preserving avg (given noise 2000):  {inc['noisy_avg']:.4f}")
    print(f"  Privacy-preserving total (given noise 5000): {inc['noisy_total']:.4f}")
    print("  Composition (epsilon1=0.05, epsilon2=0.05):")
    print(f"    b_avg (split):        {inc['b_avg_split']:.4f}")
    print(f"    b_total (split):      {inc['b_total_split']:.4f}")
    print(f"    Noisy avg (split):    {inc['noisy_avg_split']:.4f}")
    print(f"    Noisy total (split):  {inc['noisy_total_split']:.4f}")
    print(f"    Total epsilon:        {inc['epsilon_total_basic_composition']:.4f}")
    print()

    print("=" * 60)
    print("Q2 Part 2 — Counting queries")
    print("=" * 60)
    print(f"  b (base):               {cnt['b_base']:.4f}")
    print(f"  P(noisy > 505) base:   {cnt['prob_base_gt_threshold']:.6f}")
    print("  Sequential (epsilon_i = epsilon/k, delta_i = delta/k):")
    print(f"    epsilon_i:            {cnt['epsilon_i']:.6f}")
    print(f"    b (sequential):       {cnt['b_sequential']:.4f}")
    print(f"    P(noisy > 505) seq:  {cnt['prob_sequential_gt_threshold']:.6f}")
    print("  Unbounded DP (p=0.01, n=500):")
    print(f"    Delta_f unbounded:    {cnt['delta_f_unbounded']:.4f}")
    print(f"    b (unbounded):        {cnt['b_unbounded']:.4f}")
    print(f"    P(noisy > 505) unb:  {cnt['prob_unbounded_gt_threshold']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
