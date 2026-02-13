from privacy import (
    compose_epsilons,
    counting_query_results,
    income_query_results,
    laplace_cdf_threshold,
    laplace_scale,
)


def test_laplace_scale():
    assert abs(laplace_scale(1.0, 0.5) - 2.0) < 1e-12


def test_compose_epsilons():
    assert compose_epsilons([0.5, 0.7]) == 1.2


def test_laplace_cdf_threshold():
    # simple sanity: threshold far below true_value => nearly 1.0
    p = laplace_cdf_threshold(-1000, 0, 1.0, 1.0)
    assert 0.0 <= p <= 1.0


def test_income_query_results_exact_defaults():
    out = income_query_results()
    assert abs(out["b_avg"] - 50000.0) < 1e-12
    assert abs(out["b_total"] - 500000.0) < 1e-12
    assert abs(out["noisy_avg"] - 42000.0) < 1e-12
    assert abs(out["noisy_total"] - 20005000.0) < 1e-12
    assert abs(out["b_avg_split"] - 100000.0) < 1e-12
    assert abs(out["b_total_split"] - 1000000.0) < 1e-12
    assert abs(out["noisy_avg_split"] - 44000.0) < 1e-12
    assert abs(out["noisy_total_split"] - 20010000.0) < 1e-12
    assert abs(out["epsilon_total_basic_composition"] - 0.1) < 1e-12


def test_counting_query_results_exact_defaults():
    out = counting_query_results()
    assert abs(out["b_base"] - 10.0) < 1e-12
    assert abs(out["epsilon_i"] - (0.1 / 92.0)) < 1e-12
    assert abs(out["b_sequential"] - 920.0) < 1e-10
    assert abs(out["delta_f_unbounded"] - 5.0) < 1e-12
    assert abs(out["b_unbounded"] - 4600.0) < 1e-8
    assert abs(out["prob_base_gt_threshold"] - 0.3032653298563167) < 1e-12
    assert abs(out["prob_sequential_gt_threshold"] - 0.49728997955210685) < 1e-12
    assert abs(out["prob_unbounded_gt_threshold"] - 0.49945681700076194) < 1e-12
