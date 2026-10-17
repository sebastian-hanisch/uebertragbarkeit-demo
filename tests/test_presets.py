"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import utr_constants as C
import utr_evaluation as ev
import utr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Verschobene Verteilung (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(**p) and next(iter(C.PRESETS)) == "Verschobene Verteilung (Voreinstellung)"
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_two_settings():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Verschobene Verteilung (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 2, name
    assert C.PRESETS["Keine Verschiebung"]["delta"] == 0.0 and C.PRESETS["Kein Effektmodifikator"]["tau1"] == 0.0 and C.PRESETS["Zusatzwirkung im Ziel"]["gamma"] > 0 and C.PRESETS["Niveauunterschied"]["shift"] > 0
    assert C.PRESETS["Geringe Überlappung"]["delta"] > 2 and C.PRESETS["Geringe Überlappung"]["tau2"] > 0


def test_the_presets_show_what_their_help_says():
    by = {n: ev.analyse(ev.Settings(**p)) for n, p in C.PRESETS.items()}
    assert abs(by["Keine Verschiebung"].rows[0].error) < 0.7 and abs(by["Kein Effektmodifikator"].rows[0].error) < 0.7
    assert abs(by["Zusatzwirkung im Ziel"].rows[1].error + 0.76) < 0.5 and by["Zusatzwirkung im Ziel"].truth == pytest.approx(by["Verschobene Verteilung (Voreinstellung)"].truth + 1.0)
    assert by["Niveauunterschied"].level_difference > by["Verschobene Verteilung (Voreinstellung)"].level_difference + 19.0
    assert by["Geringe Überlappung"].ess_exact < 0.01 and by["Verschobene Verteilung (Voreinstellung)"].rows[0].error < -0.5


def test_snap_clamps_rounds_and_maps_the_model():
    d = P.SETTING_SPECS["delta_slider"]
    assert P.snap(9.0, d) == C.DELTA_MAX and P.snap(-1.0, d) == C.DELTA_MIN and P.snap(0.6, d) == pytest.approx(0.5) and P.snap(0.7, d) == pytest.approx(0.75)
    n = P.SETTING_SPECS["ns_slider"]
    assert P.snap(50, n) == C.NS_MIN and P.snap(10 ** 6, n) == C.NS_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    m = P.SETTING_SPECS["model_radio"]
    assert P.snap(0, m) == "linear" and P.snap(1, m) == "quadratisch" and P.snap(5, m) == "quadratisch" and P.snap(-2, m) == "linear" and P.url_value("model_radio", "quadratisch") == 1 and P.url_value("delta_slider", 0.5) == 0.5


def test_bounds_defaults_and_params():
    assert P.bounds("ns_slider") == (C.NS_MIN, C.NS_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gamma_slider") == (C.GAMMA_MIN, C.GAMMA_MAX)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"d", "t1", "t2", "g", "lv", "ns", "nt", "seed", "m"} and C.DEFAULT_NS in range(C.NS_MIN, C.NS_MAX + 1, C.NS_STEP) and C.DEFAULT_MODEL == C.MODELS[0]


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.EXP_DELTAS == tuple(sorted(C.EXP_DELTAS)) and C.EXP_TAU1S == tuple(sorted(C.EXP_TAU1S)) and 0.0 in C.EXP_DELTAS and 0.0 in C.EXP_TAU1S and 1.0 in C.EXP_DELTAS and 1.5 in C.EXP_TAU1S
    assert C.EXP_PILOTS == tuple(sorted(C.EXP_PILOTS)) and 0.0 in C.EXP_GAMMAS and 0.0 in C.EXP_SHIFTS and C.EXP_OVERLAP_DELTAS == tuple(sorted(C.EXP_OVERLAP_DELTAS)) and C.EXP_TABLES >= 1
