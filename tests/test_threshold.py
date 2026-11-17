from src.models.threshold import best_threshold, total_cost


def test_total_cost_counts_errors():
    # seuil 0,5 : un vrai positif, une panne ratée, une fausse alerte, un vrai négatif
    y = [1, 1, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1]
    assert total_cost(y, p, 0.5, cost_fn=10, cost_fp=1) == 11


def test_expensive_misses_lower_the_threshold():
    y = [1, 0, 0, 0]
    p = [0.4, 0.5, 0.6, 0.1]
    t_equal, _ = best_threshold(y, p, cost_fn=1, cost_fp=1)
    t_costly_fn, _ = best_threshold(y, p, cost_fn=10, cost_fp=1)
    assert t_costly_fn < t_equal


def test_perfect_separation_has_zero_cost():
    t, cost = best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert cost == 0
    assert 0.2 < t <= 0.8
