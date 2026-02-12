from privacy import laplace_scale, laplace_cdf_threshold, compose_epsilons


def test_laplace_scale():
    assert abs(laplace_scale(1.0, 0.5) - 2.0) < 1e-12


def test_compose_epsilons():
    assert compose_epsilons([0.5, 0.7]) == 1.2


def test_laplace_cdf_threshold():
    # simple sanity: threshold far below true_value => nearly 1.0
    p = laplace_cdf_threshold(-1000, 0, 1.0, 1.0)
    assert 0.0 <= p <= 1.0
