"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import utr_evaluation as ev


def test_instance_verdict_in_the_base_case_is_a_success():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "Die Wahrheit im Ziel ist 3,50 (in der Quelle 2,00)" in text and "Die naive Übertragung schätzt 2,27 (Fehler −1,23)" in text and "deckt die Wahrheit" in text and "42,8 %" in text


def test_instance_verdict_warns_about_an_unobserved_additional_effect():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(gamma=1.0)))
    assert kind == "warning" and "unbeobachtete Zusatzwirkung von 1,00" in text and "um sie daneben" in text


def test_instance_verdict_warns_when_the_model_does_not_fit_the_modifier():
    for seed in range(60):
        a = ev.analyse(ev.Settings(delta=2.5, tau2=0.5, ns=2000, nt=2000, seed=seed))
        if not a.rows[1].covers:
            kind, text = ev.verdict_instance(a)
            assert kind == "warning" and "Wirkungsmodell der Quelle passt hier nicht zur Form des Modifikators" in text and "Zusatzwirkung" not in text
            return
    raise AssertionError("bei starker Krümmung und linearem Modell sollte das Intervall verfehlen")


def test_gamma_is_named_before_a_model_misfit():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(delta=2.5, tau2=0.5, gamma=1.0, ns=2000, nt=2000)))
    assert kind == "warning" and "Zusatzwirkung" in text and "passt hier nicht" not in text


def test_shift_verdict(shift_res):
    kind, text = ev.verdict_shift(shift_res)
    assert kind == "info" and "Bei Δ = 1 und τ₁ = 1,5 liegt die naive Übertragung im Mittel um −1,49 daneben, die transportierte um +0,02" in text and "bei Δ = 2 und τ₁ = 3 sind es −6,00 gegen −0,04" in text
    assert "in allen 16 Zellen zwischen 93 % und 97 %" in text and "von 0,28 (Δ = 0) auf 0,63 (Δ = 2)" in text


def test_mechanism_verdict(mechanism_res):
    kind, text = ev.verdict_mechanism(mechanism_res)
    assert kind == "info" and "Liegt der Servicegrad im Ziel um 20,0 höher, bleibt die übertragene Wirkung 1,97 (Wahrheit 2,00)" in text and "Fehler −1,98" in text
    assert "Pilotstudie mit 60 Standorten widerspricht der Übertragung nur in 44 % der Läufe, mit 600 in 98 %" in text


def test_overlap_verdict(overlap_res):
    kind, text = ev.verdict_overlap(overlap_res)
    assert kind == "info" and "bei Δ = 1 nur 37 % von n, bei Δ = 2 1,8 %" in text and "lineare Modell bei Δ = 2 um −1,37 daneben, das quadratische um +0,09" in text and "streut dafür mit 1,14 gegen 0,66" in text


def test_selection_verdict(selection_res):
    kind, text = ev.verdict_selection(selection_res)
    assert kind == "info" and "Von 1024 Selektionsdiagrammen haben 512 keinen Pfeil von X nach Y" in text and "in 136 (26,6 %) direkt übertragbar" in text and "in 120 (23,4 %) nur über eine zulässige Menge" in text
    assert "in 256 (50,0 %) nicht" in text and "in 0 von 4560 Fällen von der wahren Zielwirkung ab" in text and "in 0 von 7728 Fällen trotzdem" in text
