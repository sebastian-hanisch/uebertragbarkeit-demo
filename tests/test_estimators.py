"""Schätzer: Wirkungsfunktion, naive Übertragung, transportierte und gewichtete Wirkung, effektive Stichprobe, Pilotstudie; an den Handrechnungen aus conftest.py."""

import numpy as np
import pytest

import utr_constants as C
import utr_estimators as E
import utr_world as W
from kausal_generator import SplitMix64


def test_ols_against_numpy_lstsq():
    rng = SplitMix64(1)
    x1, x2 = rng.normal(80), rng.normal(80)
    y = 1.0 + 2.0 * x1 - x2 + 0.5 * rng.normal(80)
    beta, cov = E.ols(y, [x1, x2])
    d = np.column_stack([np.ones(80), x1, x2])
    ref, res, *_ = np.linalg.lstsq(d, y, rcond=None)
    assert beta == pytest.approx(ref) and cov == pytest.approx(np.linalg.inv(d.T @ d) * res[0] / 77)


def test_effect_function_fit_on_exact_data(exact_fit):
    z, x, y = exact_fit
    fit = E.fit_effect_function(z, x, y)
    assert fit.degree == 1 and fit.coef == pytest.approx([2.0, 1.5]) and np.abs(fit.cov).max() < 1e-20
    assert fit.at([-1.0, 0.0, 1.0]) == pytest.approx([0.5, 2.0, 3.5]) and fit.at(2.0) == pytest.approx(5.0)


def test_effect_function_fit_with_a_quadratic_term(quadratic_fit):
    z, x, y = quadratic_fit
    fit = E.fit_effect_function(z, x, y, degree=2)
    assert fit.degree == 2 and fit.coef == pytest.approx([2.0, 1.5, 0.5]) and fit.at(2.0) == pytest.approx(2 + 3 + 2) and fit.at([-2.0, 0.0]) == pytest.approx([1.0, 2.0])
    linear = E.fit_effect_function(z, x, y, degree=1)
    assert linear.coef[1] == pytest.approx(1.5) and linear.coef[0] != pytest.approx(2.0)          # das lineare Modell biegt den Achsenabschnitt der Wirkung (z² hat Mittel 2)


def test_evaluate_by_hand():
    fit = E.EffectFit(1, np.array([2.0, 1.5]), np.zeros((2, 2)))
    assert E.evaluate(fit, np.array([-1.0, 1.0]))["estimate"] == pytest.approx(2.0)                       # z-Mittel 0
    assert E.evaluate(fit, np.array([1.0, 3.0]))["estimate"] == pytest.approx(2.0 + 1.5 * 2.0)            # z-Mittel 2
    r = E.evaluate(fit, np.array([0.0, 2.0]), np.array([1.0, 3.0]))                                        # Gewichte 1/4 und 3/4: Mittel 1,5
    assert r["estimate"] == pytest.approx(2.0 + 1.5 * 1.5)
    assert r["se"] == pytest.approx(1.5 * np.sqrt(0.25 ** 2 * 1.5 ** 2 + 0.75 ** 2 * 0.5 ** 2)) and r["lo"] == pytest.approx(r["estimate"] - C.Z_CRIT * r["se"]) and r["hi"] == pytest.approx(r["estimate"] + C.Z_CRIT * r["se"])


def test_evaluate_includes_the_coefficient_covariance():
    fit = E.EffectFit(1, np.array([2.0, 1.5]), np.array([[0.04, 0.0], [0.0, 0.01]]))
    r = E.evaluate(fit, np.array([2.0, 2.0]))                                                              # Mittel 2, Streuung der Momente 0
    assert r["se"] == pytest.approx(np.sqrt(0.04 + 4.0 * 0.01))
    cross = E.evaluate(E.EffectFit(1, np.array([2.0, 1.5]), np.array([[0.04, 0.005], [0.005, 0.01]])), np.array([2.0, 2.0]))
    assert cross["se"] == pytest.approx(np.sqrt(0.04 + 4.0 * 0.01 + 2 * 2.0 * 0.005))


def test_evaluate_with_a_quadratic_fit(quadratic_fit):
    z, x, y = quadratic_fit
    fit = E.fit_effect_function(z, x, y, 2)
    assert E.evaluate(fit, np.array([1.0, 3.0]))["estimate"] == pytest.approx(2 + 1.5 * 2 + 0.5 * 5)      # Momente 2 und 5
    assert E.evaluate(fit, z[:5])["estimate"] == pytest.approx(2 + 0 + 0.5 * 2)


def test_naive_transported_and_weighted_in_a_large_sample():
    w = W.World(delta=1.0, tau1=1.5)
    src, tgt = W.simulate(w, 300000, SplitMix64(2)), W.simulate(w, 300000, SplitMix64(3), target=True)
    fit = E.fit_effect_function(src["z"][0], src["x"][0], src["y"][0])
    assert E.naive_transfer(fit, src["z"][0])["estimate"] == pytest.approx(2.0, abs=0.03) and E.transported(fit, tgt["z"][0])["estimate"] == pytest.approx(3.5, abs=0.03)
    assert E.weighted(fit, src["z"][0], 1.0)["estimate"] == pytest.approx(3.5, abs=0.08)


def test_ess_by_hand():
    assert E.ess(np.ones(10)) == pytest.approx(10.0) and E.ess(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0) and E.ess(np.array([1.0, 1.0, 2.0])) == pytest.approx(16.0 / 6.0)
    assert E.ess(W.gauss_weights(SplitMix64(4).normal(400000), 1.0)) / 400000 == pytest.approx(np.exp(-1.0), abs=0.02)


def test_the_weighted_estimate_with_no_shift_is_the_naive_one():
    w = W.World(delta=0.0)
    s = W.simulate(w, 200, SplitMix64(5))
    fit = E.fit_effect_function(s["z"][0], s["x"][0], s["y"][0])
    assert E.weighted(fit, s["z"][0], 0.0) == pytest.approx(E.naive_transfer(fit, s["z"][0]))


def test_pilot_contradicts_a_wrong_transfer_and_not_a_right_one():
    w = W.World(delta=0.0, tau1=1.5)
    rng = SplitMix64(6)
    wrong = {"estimate": 50.0, "se": 0.3}
    assert all(E.pilot_contradicts(w, 200, rng, wrong) for _ in range(5))
    right = {"estimate": W.target_truth(w), "se": 0.3}
    assert np.mean([E.pilot_contradicts(w, 200, rng, right) for _ in range(200)]) < 0.12


def test_pilot_detects_the_additional_effect_more_often_with_more_sites():
    w = W.World(delta=0.0, tau1=1.5, gamma=1.0)
    t = {"estimate": W.source_truth(w), "se": 0.25}
    rates = []
    for n in (60, 600):
        rng = SplitMix64(7)
        rates.append(np.mean([E.pilot_contradicts(w, n, rng, t) for _ in range(150)]))
    assert rates[1] > rates[0] + 0.2
