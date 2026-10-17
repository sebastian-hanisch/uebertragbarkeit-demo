"""Vehikel: Wirkungsfunktion, Wahrheit je Gruppe, Stichprobe, Gewichte."""

import numpy as np
import pytest

import utr_constants as C
import utr_world as W
from kausal_generator import SplitMix64


def test_effect_function_by_hand():
    w = W.World(delta=1.0, tau1=1.5, tau2=0.5, gamma=0.75)
    assert W.effect_function(w, 0.0) == pytest.approx(2.0) and W.effect_function(w, 2.0) == pytest.approx(2 + 3 + 2) and W.effect_function(w, [-1.0, 1.0]) == pytest.approx([1.0, 4.0])
    assert W.effect_function(w, 0.0, target=True) == pytest.approx(2.75) and W.effect_function(w, [0.0, 2.0], True) == pytest.approx([2.75, 7.75])


def test_truth_per_group_by_hand():
    assert W.target_truth(W.World(delta=1.0, tau1=1.5)) == pytest.approx(3.5) and W.source_truth(W.World(delta=1.0, tau1=1.5)) == pytest.approx(2.0)
    assert W.target_truth(W.World(delta=1.0, tau1=1.0, tau2=0.5)) == pytest.approx(2 + 1 + 0.5 * 2) and W.source_truth(W.World(tau2=0.5)) == pytest.approx(2.5)
    assert W.target_truth(W.World(delta=2.0, tau1=3.0, gamma=1.0)) == pytest.approx(2 + 6 + 1) and W.target_truth(W.World(delta=0.0, tau1=3.0)) == W.source_truth(W.World(delta=0.0, tau1=3.0))


def test_truth_matches_the_average_of_the_effect_function_over_z():
    w = W.World(delta=1.2, tau1=0.75, tau2=0.5, gamma=0.25)
    z = 1.2 + SplitMix64(1).normal(1000000)
    assert W.effect_function(w, z, True).mean() == pytest.approx(W.target_truth(w), abs=0.01)
    z0 = SplitMix64(2).normal(1000000)
    assert W.effect_function(w, z0).mean() == pytest.approx(W.source_truth(w), abs=0.01)


def test_simulate_shapes_streams_and_groups():
    w = W.World()
    a = W.simulate(w, 40, SplitMix64(3), reps=4)
    assert set(a) == {"z", "x", "y"} and all(v.shape == (4, 40) for v in a.values()) and set(np.unique(a["x"])) <= {0.0, 1.0}
    b = W.simulate(w, 40, SplitMix64(3), reps=4)
    assert all((a[k] == b[k]).all() for k in a) and not (W.simulate(w, 40, SplitMix64(4), reps=4)["y"] == a["y"]).all()
    big_s, big_t = W.simulate(w, 200000, SplitMix64(5)), W.simulate(w, 200000, SplitMix64(6), target=True)
    assert big_s["z"].mean() == pytest.approx(0.0, abs=0.01) and big_t["z"].mean() == pytest.approx(1.0, abs=0.01) and big_t["z"].std() == pytest.approx(1.0, abs=0.01)


def test_assignment_rule_follows_z_relative_to_the_group_mean():
    for target in (False, True):
        s = W.simulate(W.World(delta=2.0), 300000, SplitMix64(7), target=target)
        z, x = s["z"][0], s["x"][0]
        mean = 2.0 if target else 0.0
        assert x.mean() == pytest.approx(0.5, abs=0.01) and x[z - mean > 1.0].mean() > 0.65 and x[z - mean < -1.0].mean() < 0.35


def test_randomized_assignment_is_independent_of_z():
    s = W.simulate(W.World(), 200000, SplitMix64(8), target=True, randomized=True)
    assert abs(np.corrcoef(s["z"][0], s["x"][0])[0, 1]) < 0.01 and s["x"][0].mean() == pytest.approx(0.5, abs=0.01)


def test_outcome_equation_and_the_effect_by_group():
    """Randomisiert: Differenz der Mittelwerte = mittlere Wirkung der Gruppe (Quelle und Ziel mit Zusatzwirkung)."""
    w = W.World(delta=1.0, tau1=1.5, tau2=0.5, gamma=0.5)
    for target in (False, True):
        s = W.simulate(w, 600000, SplitMix64(9), target=target, randomized=True)
        y, x = s["y"][0], s["x"][0]
        assert y[x == 1].mean() - y[x == 0].mean() == pytest.approx(W.target_truth(w) if target else W.source_truth(w), abs=0.05)


def test_level_shift_moves_the_target_level_not_the_effect():
    w0, w1 = W.World(delta=0.0, tau1=1.5), W.World(delta=0.0, tau1=1.5, shift=20.0)
    t0, t1 = W.simulate(w0, 100000, SplitMix64(10), target=True, randomized=True), W.simulate(w1, 100000, SplitMix64(10), target=True, randomized=True)
    assert t1["y"].mean() - t0["y"].mean() == pytest.approx(20.0, abs=1e-9)
    s = W.simulate(w1, 100000, SplitMix64(10), randomized=True)
    assert s["y"].mean() == pytest.approx(W.simulate(w0, 100000, SplitMix64(10), randomized=True)["y"].mean())


def test_gauss_weights_by_hand_and_in_expectation():
    assert W.gauss_weights(np.array([0.0]), 1.0) == pytest.approx(np.exp(-0.5)) and W.gauss_weights(np.array([2.0]), 1.0) == pytest.approx(np.exp(1.5)) and W.gauss_weights(np.array([3.0]), 0.0) == pytest.approx(1.0)
    z = SplitMix64(11).normal(2000000)
    assert W.gauss_weights(z, 1.0).mean() == pytest.approx(1.0, abs=0.01) and (W.gauss_weights(z, 1.0) * z).mean() == pytest.approx(1.0, abs=0.02)         # E[w] = 1, E[w z] = Δ


def test_exact_ess_ratio():
    assert W.exact_ess_ratio(0.0) == 1.0 and W.exact_ess_ratio(1.0) == pytest.approx(0.3678794) and W.exact_ess_ratio(2.0) == pytest.approx(np.exp(-4.0))
    assert C.DEFAULT_DELTA == 1.0
