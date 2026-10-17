"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import utr_constants as C
import utr_selection as Sel

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("delta_slider", "tau1_slider", "tau2_slider", "gamma_slider", "shift_slider", "ns_slider", "nt_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung im Ziel"] == "3,50" and m["Naive Übertragung"] == "+2,27" and m["Transportiert (Z im Ziel gemessen)"] == "+3,74" and m["Gewichtet (bekannte Dichten)"] == "+3,68"
    assert at.slider(key="delta_slider").value == C.DEFAULT_DELTA and at.radio(key="model_radio").value == C.DEFAULT_MODEL and at.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert any("Die Wahrheit im Ziel ist 3,50" in s.value for s in at.success)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # der Versuch auf Knopfdruck läuft im eigenen Test
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        for key, state_key in (("delta", "delta_slider"), ("tau1", "tau1_slider"), ("tau2", "tau2_slider"), ("gamma", "gamma_slider"), ("shift", "shift_slider"), ("ns", "ns_slider"), ("nt", "nt_slider")):
            assert at.slider(key=state_key).value == preset[key], (name, key)
        assert at.radio(key="model_radio").value == preset["model"]


def test_gamma_preset_shows_the_warning_and_the_model_radio_works():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Zusatzwirkung im Ziel")].click().run(timeout=TIMEOUT)
    assert any("unbeobachtete Zusatzwirkung" in w.value for w in at.warning)
    at.radio(key="model_radio").set_value(C.MODELS[1]).run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="model_radio").value == C.MODELS[1]


def test_selection_builder_default_and_examples():
    at = _fresh_app()
    assert at.checkbox(key="sel_s_Z").value and not at.checkbox(key="sel_s_Y").value and any("nur über eine Menge übertragbar" in s.value for s in at.success)
    labels = [b.label for b in at.button]
    for name, expected in (("S → Y: anderer Mechanismus", "nicht übertragbar"), ("S → X: andere Zuteilung", "direkt übertragbar"), ("S → W: W ohne Weg zu Y", "direkt übertragbar")):
        at.button[labels.index(name)].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert any(expected in x.value for x in list(at.success) + list(at.error)), name
    assert at.checkbox(key="sel_s_W").value is True


def test_selection_builder_checkbox_toggle_changes_the_verdict():
    at = _fresh_app()
    at.checkbox(key="sel_s_Y").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception and any("nicht übertragbar" in e.value for e in at.error)
    at.checkbox(key="sel_edge_XY").set_value(False).run(timeout=TIMEOUT)
    assert not at.exception and any("keinen Pfeil von X nach Y" in i.value for i in at.info)


def test_the_selection_experiment_runs_on_click():
    at = _fresh_app()
    assert "selection_ready" not in at.session_state and not any("Von 1024 Selektionsdiagrammen" in i.value for i in at.info)
    at.button(key="selection_run").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Von 1024 Selektionsdiagrammen haben 512 keinen Pfeil von X nach Y" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["d"] = "9"
    at.query_params["t1"] = "0.9"
    at.query_params["ns"] = "50"
    at.query_params["m"] = "1"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="delta_slider").value == C.DELTA_MAX and at.slider(key="tau1_slider").value == 1.0 and at.slider(key="ns_slider").value == C.NS_MIN
    assert at.radio(key="model_radio").value == C.MODELS[1] and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="gamma_slider").set_value(0.5).run(timeout=TIMEOUT)
    assert at.query_params["g"] in ("0.5", ["0.5"])
    at.radio(key="model_radio").set_value(C.MODELS[1]).run(timeout=TIMEOUT)
    assert at.query_params["m"] in ("1", ["1"])
    assert Sel.NAMES[Sel.S] == "S"
