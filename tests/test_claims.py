"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil) oder sind exakte Rechnungen; Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle. Die Verzerrung der naiven Übertragung (−τ₁ Δ) ist eine geschlossene Form, kein Befund; der Befund ist, dass die Umgewichtung sie vollständig
beseitigt und was sie kostet. Die 512 Diagramme ohne Pfeil X → Y sind Setup (die Wirkung ist überall null), nicht Befund.
"""

import numpy as np
import pytest

import utr_constants as C
import utr_evaluation as ev


def test_default_sample():
    a = ev.analyse(ev.Settings())
    n, t, w = a.rows
    assert a.truth == 3.5 and a.source_truth == 2.0 and n.estimate == pytest.approx(2.27, abs=0.005) and n.error == pytest.approx(-1.23, abs=0.005) and n.covers is False
    assert t.estimate == pytest.approx(3.74, abs=0.005) and t.error == pytest.approx(0.24, abs=0.005) and t.se == pytest.approx(0.421, abs=0.0006) and t.covers is True and w.estimate == pytest.approx(3.68, abs=0.005)
    assert a.ess_ratio == pytest.approx(0.428, abs=0.0006) and a.ess_exact == pytest.approx(0.368, abs=0.0006) and ev.verdict_instance(a)[0] == "success"


def test_hand_values_of_the_text():
    import utr_selection as Sel
    import utr_world as W

    assert W.target_truth(W.World(delta=1.0, tau1=1.5)) == 3.5 and W.source_truth(W.World(delta=1.0, tau1=1.5)) == 2.0 and W.target_truth(W.World(delta=1.0, tau1=1.0, tau2=0.5)) == 4.0
    assert W.exact_ess_ratio(1.0) == pytest.approx(0.3679, abs=0.00005)
    parents = Sel.parents_of((1 << Sel.EDGES.index((Sel.Z, Sel.Y))) | (1 << Sel.EDGES.index((Sel.X, Sel.Y))), 1)
    src, tgt = {Sel.W: [0.5], Sel.X: [0.5], Sel.Y: [0.3, 0.4, 0.5, 0.8], Sel.Z: [0.2]}, {Sel.W: [0.5], Sel.X: [0.5], Sel.Y: [0.3, 0.4, 0.5, 0.8], Sel.Z: [0.7]}
    assert Sel.transport_formula(parents, src, tgt, ()) == pytest.approx(0.14) and Sel.transport_formula(parents, src, tgt, (Sel.Z,)) == pytest.approx(0.24) and Sel.target_effect(parents, tgt) == pytest.approx(0.24)


def test_experiment_1_shift_and_modifier(shift_res):
    by = {(r["delta"], r["tau1"]): r for r in shift_res}
    assert len(shift_res) == 16 and C.EXP_N == 500 and C.EXP_NT == 500 and C.EXP_REPS == 300
    for key, naive in (((0.5, 0.75), -0.37), ((0.5, 1.5), -0.75), ((0.5, 3.0), -1.50), ((1.0, 0.75), -0.75), ((1.0, 1.5), -1.49), ((1.0, 3.0), -3.00), ((2.0, 0.75), -1.50), ((2.0, 1.5), -3.00), ((2.0, 3.0), -6.00)):
        assert by[key]["naive_bias"] == pytest.approx(naive, abs=0.005), key
        assert by[key]["naive_bias"] == pytest.approx(-key[1] * key[0], abs=0.02), key                          # geschlossene Form −τ₁ Δ
    assert max(abs(r["transported_bias"]) for r in shift_res) < 0.04 and all(abs(by[(d, 0.0)]["naive_bias"]) < 0.02 for d in (0.0, 0.5, 1.0, 2.0)) and all(abs(by[(0.0, t)]["naive_bias"]) < 0.02 for t in (0.0, 0.75, 1.5, 3.0))
    assert min(r["cover"] for r in shift_res) == pytest.approx(0.927, abs=0.0006) and max(r["cover"] for r in shift_res) == pytest.approx(0.967, abs=0.0006)
    assert by[(0.0, 0.0)]["transported_sd"] == pytest.approx(0.281, abs=0.0006) and by[(1.0, 1.5)]["transported_sd"] == pytest.approx(0.431, abs=0.0006) and by[(2.0, 3.0)]["transported_sd"] == pytest.approx(0.626, abs=0.0006)
    assert by[(2.0, 3.0)]["weighted_sd"] == pytest.approx(1.153, abs=0.0006) and by[(2.0, 3.0)]["weighted_bias"] == pytest.approx(-0.273, abs=0.0006)


def test_experiment_2_level_and_additional_effect(mechanism_res):
    lv = {r["shift"]: r for r in mechanism_res["level"]}
    assert lv[0.0]["transferred"] == pytest.approx(2.01, abs=0.005) and lv[5.0]["transferred"] == pytest.approx(2.01, abs=0.005) and lv[20.0]["transferred"] == pytest.approx(1.97, abs=0.005)
    assert lv[5.0]["level_difference"] == pytest.approx(5.00, abs=0.005) and lv[20.0]["level_difference"] == pytest.approx(19.98, abs=0.005) and lv[0.0]["truth"] == 2.0
    g = {r["gamma"]: r for r in mechanism_res["gammas"]}
    for gamma, err, cover in ((0.0, -0.02, 0.95), (0.5, -0.49, 0.613), (1.0, -0.98, 0.09), (2.0, -1.98, 0.0)):
        assert g[gamma]["error"] == pytest.approx(err, abs=0.005) and g[gamma]["cover"] == pytest.approx(cover, abs=0.0006), gamma
    for gamma, pilots in ((0.0, (0.093, 0.043, 0.047)), (0.5, (0.070, 0.100, 0.153)), (1.0, (0.157, 0.327, 0.580)), (2.0, (0.437, 0.860, 0.983))):
        assert [g[gamma]["pilot"][p] for p in C.EXP_PILOTS] == pytest.approx(list(pilots), abs=0.0006), gamma


def test_experiment_3_overlap_and_extrapolation(overlap_res):
    by = {r["delta"]: r for r in overlap_res}
    assert [round(by[d]["ess_exact"], 3) for d in (0.5, 1.0, 2.0, 3.0)] == [0.779, 0.368, 0.018, 0.000]
    assert [round(by[d]["ess_sample"], 3) for d in (0.5, 1.0, 2.0, 3.0)] == [0.779, 0.369, 0.020, 0.001] and [round(by[d]["truth"], 3) for d in (0.5, 1.0, 2.0, 3.0)] == [3.125, 4.0, 6.5, 10.0]
    for d, lb, ls, qb, qs in ((0.5, -0.09, 0.33, -0.02, 0.36), (1.0, -0.29, 0.44, -0.06, 0.56), (2.0, -1.37, 0.66, 0.09, 1.14), (3.0, -3.50, 0.95, -0.07, 2.28)):
        r = by[d]
        assert (r["linear_bias"], r["linear_sd"], r["quadratic_bias"], r["quadratic_sd"]) == pytest.approx((lb, ls, qb, qs), abs=0.005), d


def test_experiment_4_selection_diagrams(selection_res):
    r = selection_res
    assert r["diagrams"] == 1024 and r["kinds"] == {"trivial": 512, "direkt": 136, "menge": 120, "keine": 256} and r["tables"] == 6
    assert r["kinds"]["direkt"] / 512 == pytest.approx(0.266, abs=0.0005) and r["kinds"]["menge"] / 512 == pytest.approx(0.234, abs=0.0005) and r["kinds"]["keine"] / 512 == 0.5
    assert r["checked_true"] == 4560 and r["wrong_true"] == 0 and r["checked_false"] == 7728 and r["equal_although_false"] == 0
    assert {k: (v["diagrams"], v["direkt"], v["menge"], v["keine"]) for k, v in r["by_s"].items()} == {0: (32, 32, 0, 0), 1: (128, 60, 36, 32), 2: (192, 36, 60, 96), 3: (128, 8, 24, 96), 4: (32, 0, 0, 32)}


def test_the_text_numbers_of_the_app_follow_the_findings(shift_res, mechanism_res, overlap_res, selection_res):
    assert "−1,49 daneben, die transportierte um +0,02" in ev.verdict_shift(shift_res)[1] and "zwischen 93 % und 97 %" in ev.verdict_shift(shift_res)[1]
    assert "Fehler −1,98" in ev.verdict_mechanism(mechanism_res)[1] and "mit 600 in 98 %" in ev.verdict_mechanism(mechanism_res)[1]
    assert "Δ = 2 1,8 %" in ev.verdict_overlap(overlap_res)[1] and "(26,6 %) direkt übertragbar" in ev.verdict_selection(selection_res)[1]
    assert np.exp(-1.0) == pytest.approx(0.368, abs=0.0006)
