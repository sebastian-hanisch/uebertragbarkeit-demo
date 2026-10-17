"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import numpy as np
import pytest

import utr_constants as C
import utr_evaluation as ev
import utr_selection as Sel
import utr_visualization as V


def test_effect_chart_has_the_three_curves_and_the_two_distributions():
    a = ev.analyse(ev.Settings(tau2=0.5, gamma=1.0))
    fig = V.build_effect_chart(a)
    names = [t.name for t in fig.data]
    assert names == ["Verteilung von Z in der Quelle", "Verteilung von Z im Ziel", "wahre Wirkung (Quelle)", "wahre Wirkung (Ziel)", "geschätzt aus der Quelle (linear)"]
    z = np.array(fig.data[2].x)
    assert fig.data[3].y == pytest.approx(np.array(fig.data[2].y) + 1.0) and fig.data[4].y == pytest.approx(a.fit.at(z))
    assert z[0] == pytest.approx(-3.0) and z[-1] == pytest.approx(3.0 + a.settings.delta)


def test_estimates_chart_has_three_points_and_both_truth_lines():
    a = ev.analyse(ev.Settings())
    fig = V.build_estimates_chart(a)
    assert len([t for t in fig.data if t.mode == "markers"]) == 3 and fig.layout.shapes[0].x0 == pytest.approx(a.truth) and fig.layout.shapes[1].x0 == pytest.approx(a.source_truth)
    colors = [t.marker.color for t in fig.data]
    assert colors == [C.COLOR_GOOD if r.covers else C.COLOR_BAD for r in a.rows]


def test_tables_by_hand():
    a = ev.analyse(ev.Settings())
    t = V.build_estimates_table(a)
    assert list(t["Schätzer"]) == list(ev.ROW_NAMES) and len(V.build_summary_table(a)) == 6 and "exp(−Δ²)" in V.build_summary_table(a).iloc[3]["Größe"]
    assert V.build_summary_table(a).iloc[1]["Wert"] == "3,50" and V.build_summary_table(a).iloc[0]["Wert"] == "2,00"
    V.style_estimates_table(t)


def test_experiment_figures_and_tables_build(shift_res, mechanism_res, overlap_res, selection_res):
    assert V.build_shift_chart(shift_res) is not None and len(V.build_shift_table(shift_res)) == 16
    assert V.build_mechanism_chart(mechanism_res) is not None and len(V.build_level_table(mechanism_res)) == 3 and len(V.build_gamma_table(mechanism_res)) == 4
    assert V.build_overlap_chart(overlap_res) is not None and len(V.build_overlap_table(overlap_res)) == 4
    assert V.build_selection_chart(selection_res) is not None and len(V.build_selection_table(selection_res)) == 5


def test_shift_table_by_hand(shift_res):
    t = V.build_shift_table(shift_res)
    row = t[(t["Δ"] == "1,0") & (t["τ₁"] == "1,50")].iloc[0]
    assert row["Wahrheit im Ziel"] == "3,50" and row["naiv: Verzerrung"] == "−1,49" and row["transportiert: Deckung"] == "93 %"


def test_gamma_table_has_one_column_per_pilot_size(mechanism_res):
    t = V.build_gamma_table(mechanism_res)
    assert [c for c in t.columns if c.startswith("Pilotstudie")] == [f"Pilotstudie {p}: widerspricht" for p in C.EXP_PILOTS] and list(t["Zusatzwirkung γ"]) == ["0,0", "0,5", "1,0", "2,0"]


def test_selection_diagram_draws_every_arrow_and_marks_the_set():
    parents = Sel.parents_of(0b100101 | 0, 0b0011)
    fig = V.build_selection_diagram(parents, (Sel.Z,))
    arrows = sum(1 for ann in fig.layout.annotations if ann.showarrow)
    assert arrows == sum(len(ps) for ps in parents) and len([t for t in fig.data if t.mode == "markers+text"]) == 5
    widths = {t.text[0]: t.marker.line.width for t in fig.data}
    assert widths["Z"] == 6 and widths["W"] == 2 and widths["X"] == 4 and widths["Y"] == 4
    s_node = next(t for t in fig.data if t.text[0] == "S")
    assert s_node.marker.symbol == "diamond" and s_node.marker.line.color == C.COLOR_BAD


def test_selection_diagram_builds_for_every_example():
    for em in range(0, 64, 7):
        for sm in range(16):
            assert V.build_selection_diagram(Sel.parents_of(em, sm)) is not None
