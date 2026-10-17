"""Auswertung: Einstellungen, Quelle und Ziel mit allen Schätzern und die vier Experimente."""

import numpy as np
import pytest

import utr_constants as C
import utr_evaluation as ev
import utr_world as W


def test_number_formats():
    assert ev.de(-1.234) == "−1,23" and ev.de(0.5, 1) == "0,5" and ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(1e-12) == "+0,00" and ev.pct(0.428) == "43 %" and ev.pct(0.4281, 1) == "42,8 %"


def test_settings_defaults_world_and_degree():
    s = ev.Settings()
    assert (s.delta, s.tau1, s.tau2, s.gamma, s.shift, s.ns, s.nt, s.seed, s.model) == (1.0, 1.5, 0.0, 0.0, 0.0, 500, 500, 35, "linear") and s.world == W.World(1.0, 1.5, 0.0, 0.0, 0.0) and s.degree == 1
    assert ev.Settings(model="quadratisch").degree == 2


def test_analyse_invariants():
    a = ev.analyse(ev.Settings())
    assert [r.name for r in a.rows] == list(ev.ROW_NAMES) and a.truth == pytest.approx(3.5) and a.source_truth == pytest.approx(2.0) and a.ess_exact == pytest.approx(np.exp(-1.0))
    for r in a.rows:
        assert r.lo == pytest.approx(r.estimate - C.Z_CRIT * r.se) and r.hi == pytest.approx(r.estimate + C.Z_CRIT * r.se) and r.error == pytest.approx(r.estimate - 3.5) and r.covers == (r.lo <= 3.5 <= r.hi)
    assert 0.2 < a.ess_ratio < 0.7 and a.fit.degree == 1 and a.source_mean_z == pytest.approx(0.0, abs=0.15) and a.target_mean_z == pytest.approx(1.0, abs=0.15)


def test_the_three_rows_agree_without_a_shift():
    a = ev.analyse(ev.Settings(delta=0.0, tau1=1.5))
    by = {r.name: r for r in a.rows}
    assert by[ev.ROW_NAMES[2]].estimate == pytest.approx(by[ev.ROW_NAMES[0]].estimate) and abs(by[ev.ROW_NAMES[1]].estimate - by[ev.ROW_NAMES[0]].estimate) < 0.2 and a.ess_ratio == pytest.approx(1.0)


def test_naive_transfer_ignores_the_target_and_the_other_rows_use_it():
    a, b = ev.analyse(ev.Settings(delta=1.0)), ev.analyse(ev.Settings(delta=2.0))
    assert a.rows[0].estimate == pytest.approx(b.rows[0].estimate, abs=0.01) or abs(a.rows[0].estimate - b.rows[0].estimate) < 0.5       # Quelle gleich (gleicher Seed), nur die Wirkungsfunktion der Daten ändert sich nicht
    assert b.rows[1].estimate > a.rows[1].estimate + 0.5 and b.truth > a.truth


def test_gamma_moves_the_truth_but_not_the_estimates():
    a, b = ev.analyse(ev.Settings(gamma=0.0)), ev.analyse(ev.Settings(gamma=1.0))
    assert b.truth == pytest.approx(a.truth + 1.0) and [r.estimate for r in a.rows] == pytest.approx([r.estimate for r in b.rows])


def test_level_shift_changes_neither_truth_nor_estimates():
    a, b = ev.analyse(ev.Settings(shift=0.0)), ev.analyse(ev.Settings(shift=20.0))
    assert a.truth == b.truth and [r.estimate for r in a.rows] == pytest.approx([r.estimate for r in b.rows]) and b.level_difference == pytest.approx(a.level_difference + 20.0, abs=1e-9)


def test_quadratic_model_fits_a_quadratic_modifier_better_at_a_large_shift():
    lin = ev.analyse(ev.Settings(delta=2.5, tau2=0.5, ns=2000, nt=2000, model="linear"))
    quad = ev.analyse(ev.Settings(delta=2.5, tau2=0.5, ns=2000, nt=2000, model="quadratisch"))
    assert abs(quad.rows[1].error) < abs(lin.rows[1].error) and quad.fit.degree == 2 and len(quad.fit.coef) == 3


def test_analyse_is_deterministic_and_seed_dependent():
    assert ev.analyse(ev.Settings()).rows[1].estimate == ev.analyse(ev.Settings()).rows[1].estimate and ev.analyse(ev.Settings(seed=1)).rows[1].estimate != ev.analyse(ev.Settings(seed=2)).rows[1].estimate


def test_shift_experiment_small_run():
    rows = ev.shift_experiment(deltas=(0.0, 1.0), tau1s=(0.0, 1.5), n=300, nt=300, reps=60)
    by = {(r["delta"], r["tau1"]): r for r in rows}
    assert len(rows) == 4 and by[(1.0, 1.5)]["truth"] == pytest.approx(3.5) and by[(1.0, 1.5)]["naive_bias"] == pytest.approx(-1.5, abs=0.15) and abs(by[(1.0, 1.5)]["transported_bias"]) < 0.15
    assert abs(by[(0.0, 0.0)]["naive_bias"]) < 0.25 and abs(by[(1.0, 0.0)]["naive_bias"]) < 0.25 and by[(1.0, 1.5)]["cover"] > 0.85
    assert rows == ev.shift_experiment(deltas=(0.0, 1.0), tau1s=(0.0, 1.5), n=300, nt=300, reps=60)


def test_mechanism_experiment_small_run():
    res = ev.mechanism_experiment(n=300, reps=60)
    assert [r["shift"] for r in res["level"]] == list(C.EXP_SHIFTS) and all(abs(r["transferred"] - 2.0) < 0.25 for r in res["level"]) and res["level"][2]["level_difference"] == pytest.approx(20.0, abs=1.0)
    g = {r["gamma"]: r for r in res["gammas"]}
    assert g[2.0]["error"] == pytest.approx(-2.0, abs=0.2) and g[2.0]["cover"] < 0.1 and g[0.0]["cover"] > 0.85 and set(g[1.0]["pilot"]) == set(C.EXP_PILOTS) and g[2.0]["pilot"][600] > g[2.0]["pilot"][60]


def test_overlap_experiment_small_run():
    rows = ev.overlap_experiment(n=300, reps=40)
    assert [r["delta"] for r in rows] == list(C.EXP_OVERLAP_DELTAS) and rows[1]["ess_exact"] == pytest.approx(np.exp(-1.0)) and rows[1]["ess_sample"] == pytest.approx(np.exp(-1.0), abs=0.02)
    assert rows[2]["linear_bias"] < -0.5 and abs(rows[2]["quadratic_bias"]) < 0.5 and rows[2]["quadratic_sd"] > rows[2]["linear_sd"]


def test_selection_experiment_small_run():
    res = ev.selection_experiment(tables=1)
    assert res["diagrams"] == 1024 and res["kinds"] == {"trivial": 512, "direkt": 136, "menge": 120, "keine": 256} and res["wrong_true"] == 0 and res["equal_although_false"] == 0
    assert list(res["by_s"]) == [0, 1, 2, 3, 4] and res["by_s"][0] == {"diagrams": 32, "direkt": 32, "menge": 0, "keine": 0} and sum(r["diagrams"] for r in res["by_s"].values()) == 512
