"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import utr_constants as C
import utr_selection as Sel
import utr_world as W
from utr_evaluation import ROW_NAMES, Analysis, Settings, de, pct, signed

SHORT = {ROW_NAMES[0]: "naiv", ROW_NAMES[1]: "transportiert", ROW_NAMES[2]: "gewichtet"}
COLORS = {ROW_NAMES[0]: C.COLOR_NEUTRAL, ROW_NAMES[1]: C.COLOR_GOOD, ROW_NAMES[2]: C.COLOR_CONTROL}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung mit 95-%-Intervall je Schätzer (grün: deckt die Wahrheit im Ziel, rot: deckt nicht); senkrecht die Wahrheit im Ziel und die in der Quelle."""
    fig = go.Figure()
    for r in a.rows:
        color = C.COLOR_GOOD if r.covers else C.COLOR_BAD
        fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=color, size=11), showlegend=False,
                                 error_x=dict(type="data", array=[r.hi - r.estimate], arrayminus=[r.estimate - r.lo], color=color, thickness=2),
                                 hovertemplate=f"{r.name}: {de(r.estimate, 2)} [{de(r.lo, 2)}; {de(r.hi, 2)}]<extra></extra>"))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit im Ziel {de(a.truth, 2)}", annotation_position="top")
    fig.add_vline(x=a.source_truth, line=dict(color="#888888", width=1, dash="dot"), annotation_text=f"Quelle {de(a.source_truth, 2)}", annotation_position="bottom")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[SHORT[n] for n in ROW_NAMES])
    fig.update_xaxes(title="mittlere Wirkung des hohen Faktors im Ziel (Punkte)")
    return _fixed(fig, 300)


def build_effect_chart(a: Analysis) -> go.Figure:
    """Die Wirkungsfunktion: wahr in der Quelle, wahr im Ziel (mit der Zusatzwirkung) und geschätzt aus der Quelle; darunter die beiden Verteilungen von Z."""
    s = a.settings
    w = s.world
    grid = np.linspace(-3.0, 3.0 + s.delta, 120)
    fig = go.Figure()
    for mean, label, color in ((0.0, "Verteilung von Z in der Quelle", C.COLOR_NEUTRAL), (s.delta, "Verteilung von Z im Ziel", C.COLOR_TRUTH)):
        dens = np.exp(-0.5 * (grid - mean) ** 2) / np.sqrt(2 * np.pi)
        fig.add_trace(go.Scatter(x=grid, y=dens * 4, mode="lines", line=dict(color=color, width=1), fill="tozeroy", opacity=0.25, name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid, y=W.effect_function(w, grid), mode="lines", line=dict(color=C.COLOR_NEUTRAL, width=3, dash="dash"), name="wahre Wirkung (Quelle)"))
    fig.add_trace(go.Scatter(x=grid, y=W.effect_function(w, grid, True), mode="lines", line=dict(color=C.COLOR_TRUTH, width=3), name="wahre Wirkung (Ziel)"))
    fig.add_trace(go.Scatter(x=grid, y=a.fit.at(grid), mode="lines", line=dict(color=C.COLOR_GOOD, width=3), name=f"geschätzt aus der Quelle ({s.model})"))
    fig.update_xaxes(title="Nachfrageschwankung Z")
    fig.update_yaxes(title="Wirkung des hohen Faktors (Punkte)")
    return _fixed(fig, 380)


def build_estimates_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Schätzer": r.name, "Schätzung": signed(r.estimate, 3), "Fehler gegen das Ziel": signed(r.error, 3), "Standardfehler": de(r.se, 3), "95-%-Intervall": f"{de(r.lo, 2)} bis {de(r.hi, 2)}",
                          "Deckt": "ja" if r.covers else "nein"} for r in a.rows])


def style_estimates_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Deckt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    return pd.DataFrame([
        {"Größe": "Mittlere Wirkung in der Quelle (wahr)", "Wert": de(a.source_truth, 2)},
        {"Größe": "Mittlere Wirkung im Ziel (wahr)", "Wert": de(a.truth, 2)},
        {"Größe": "Mittel von Z in Quelle und Ziel (gemessen)", "Wert": f"{de(a.source_mean_z, 2)} und {de(a.target_mean_z, 2)}"},
        {"Größe": "Effektive Stichprobe der Gewichte (Stichprobe / exakt exp(−Δ²))", "Wert": f"{pct(a.ess_ratio, 1)} / {pct(a.ess_exact, 1)}"},
        {"Größe": "Unterschied des mittleren Servicegrads Ziel − Quelle", "Wert": de(a.level_difference, 2)},
        {"Größe": f"Wirkungsfunktion der Quelle ({s.model}): Koeffizienten", "Wert": ", ".join(de(c, 2) for c in a.fit.coef)},
    ])


# ----------------------------------------------------------------------------- Experiment 1

def build_shift_chart(rows: list) -> go.Figure:
    labels = [f"Δ = {de(r['delta'], 1)}, τ₁ = {de(r['tau1'], 2)}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["naive_bias"] for r in rows], name="naive Übertragung", marker_color=C.COLOR_NEUTRAL))
    fig.add_trace(go.Bar(x=labels, y=[r["transported_bias"] for r in rows], name="transportiert", marker_color=C.COLOR_GOOD))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Verzerrung (Mittel minus Wahrheit im Ziel)")
    fig.update_xaxes(tickangle=-45)
    return _fixed(fig, 460)


def build_shift_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"Δ": de(r["delta"], 1), "τ₁": de(r["tau1"], 2), "Wahrheit im Ziel": de(r["truth"], 2), "naiv: Verzerrung": signed(r["naive_bias"], 2), "transportiert: Verzerrung": signed(r["transported_bias"], 2),
                          "transportiert: Streuung": de(r["transported_sd"], 2), "transportiert: Deckung": pct(r["cover"]), "gewichtet: Verzerrung": signed(r["weighted_bias"], 2),
                          "gewichtet: Streuung": de(r["weighted_sd"], 2)} for r in rows])


# ----------------------------------------------------------------------------- Experiment 2

def build_mechanism_chart(res: dict) -> go.Figure:
    g = res["gammas"]
    fig = go.Figure()
    for p in C.EXP_PILOTS:
        fig.add_trace(go.Scatter(x=[r["gamma"] for r in g], y=[100 * r["pilot"][p] for r in g], mode="lines+markers", name=f"Pilotstudie mit {p} Standorten", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[r["gamma"] for r in g], y=[100 * r["cover"] for r in g], mode="lines+markers", name="Deckung der Übertragung", line=dict(color=C.COLOR_BAD, width=3, dash="dash")))
    fig.add_hline(y=5.0, line=dict(color="#888888", width=1, dash="dot"), annotation_text="5 %", annotation_position="right")
    fig.update_xaxes(title="unbeobachtete Zusatzwirkung γ im Ziel")
    fig.update_yaxes(title="Anteil der Läufe (%)", range=[-2, 104])
    return _fixed(fig, 380)


def build_level_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Niveauunterschied im Ziel": de(r["shift"], 0), "übertragene Wirkung (Mittel)": de(r["transferred"], 2), "wahre Wirkung im Ziel": de(r["truth"], 2),
                          "Unterschied des mittleren Servicegrads": de(r["level_difference"], 2)} for r in res["level"]])


def build_gamma_table(res: dict) -> pd.DataFrame:
    out = []
    for r in res["gammas"]:
        row = {"Zusatzwirkung γ": de(r["gamma"], 1), "wahre Wirkung im Ziel": de(r["truth"], 2), "Fehler der Übertragung": signed(r["error"], 2), "Deckung": pct(r["cover"])}
        for p in C.EXP_PILOTS:
            row[f"Pilotstudie {p}: widerspricht"] = pct(r["pilot"][p])
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Experiment 3

def build_overlap_chart(rows: list) -> go.Figure:
    fig = go.Figure()
    deltas = [r["delta"] for r in rows]
    fig.add_trace(go.Bar(x=[de(d, 1) for d in deltas], y=[r["linear_bias"] for r in rows], name="lineares Modell: Verzerrung", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=[de(d, 1) for d in deltas], y=[r["quadratic_sd"] for r in rows], name="quadratisches Modell: Streuung", marker_color=C.COLOR_CONTROL))
    fig.add_trace(go.Bar(x=[de(d, 1) for d in deltas], y=[r["linear_sd"] for r in rows], name="lineares Modell: Streuung", marker_color=C.COLOR_NEUTRAL))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Verschiebung Δ der Zielgruppe (gekrümmter Modifikator τ₂ = 0,5)")
    fig.update_yaxes(title="Punkte")
    return _fixed(fig, 380)


def build_overlap_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame([{"Δ": de(r["delta"], 1), "Wahrheit im Ziel": de(r["truth"], 2), "effektive Stichprobe exakt exp(−Δ²)": pct(r["ess_exact"], 2), "aus großer Stichprobe": pct(r["ess_sample"], 2),
                          "linear: Verzerrung / Streuung": f"{signed(r['linear_bias'], 2)} / {de(r['linear_sd'], 2)}", "quadratisch: Verzerrung / Streuung": f"{signed(r['quadratic_bias'], 2)} / {de(r['quadratic_sd'], 2)}",
                          "gewichtet (quadratisch): Streuung": de(r["weighted_sd"], 2)} for r in rows])


# ----------------------------------------------------------------------------- Experiment 4 und das Selektionsdiagramm zum Bauen

def build_selection_chart(res: dict) -> go.Figure:
    labels = [f"{k} S-Pfeile" for k in res["by_s"]]
    fig = go.Figure()
    for key, label, color in (("direkt", "direkt übertragbar", C.COLOR_GOOD), ("menge", "nur über eine zulässige Menge", C.COLOR_CONTROL), ("keine", "nicht übertragbar", C.COLOR_BAD)):
        fig.add_trace(go.Bar(x=labels, y=[100 * r[key] / r["diagrams"] for r in res["by_s"].values()], name=label, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Anteil der Diagramme mit Pfeil X → Y (%)")
    return _fixed(fig, 380)


def build_selection_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"S-Pfeile": k, "Diagramme mit Pfeil X → Y": r["diagrams"], "direkt": r["direkt"], "nur über Menge": r["menge"], "nicht": r["keine"]} for k, r in res["by_s"].items()])


POS = {Sel.S: (0.0, 1.0), Sel.Z: (0.5, 1.0), Sel.W: (1.0, 1.0), Sel.X: (0.25, 0.0), Sel.Y: (0.75, 0.0)}


def build_selection_diagram(parents: tuple, adjustment=()) -> go.Figure:
    """Das Selektionsdiagramm: S (Raute, rot gestrichelte Pfeile) zeigt auf die Größen, in denen sich Quelle und Ziel unterscheiden; X grün, Y orange umrandet; Knoten der Menge dick."""
    fig = go.Figure()
    for child, ps in enumerate(parents):
        for p in ps:
            (x0, y0), (x1, y1) = POS[p], POS[child]
            dx, dy = x1 - x0, y1 - y0
            ln = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / ln, dy / ln
            color = C.COLOR_BAD if p == Sel.S else "#6b6b6b"
            fig.add_annotation(x=x1 - ux * 0.11, y=y1 - uy * 0.11, ax=x0 + ux * 0.11, ay=y0 + uy * 0.11, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3,
                               arrowwidth=3 if p != Sel.S else 2.5, arrowcolor=color, text="")
    for v in range(5):
        is_s = v == Sel.S
        color = C.COLOR_BAD if is_s else C.COLOR_GOOD if v == Sel.X else C.COLOR_TRUTH if v == Sel.Y else "#1b1b1b"
        width = 6 if v in adjustment else (4 if v in (Sel.X, Sel.Y) else 2)
        fig.add_trace(go.Scatter(x=[POS[v][0]], y=[POS[v][1]], mode="markers+text", text=[Sel.NAMES[v]], textposition="middle center", textfont=dict(size=12, color="#1b1b1b" if is_s else None) if is_s else dict(size=12),
                                 marker=dict(size=64, symbol="diamond" if is_s else "circle", color="rgba(192,57,43,0.18)" if is_s else "rgba(47,107,101,0.12)", line=dict(width=width, color=color)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.3, 1.3])
    return _fixed(fig, 340, showlegend=False)


def classify_text(parents: tuple) -> tuple[str, str, list]:
    """(Art, Text, offene Pfade) für ein Selektionsdiagramm."""
    c = Sel.classify(parents)
    sets = c["admissible"]
    names = lambda a: "{" + ", ".join(Sel.NAMES[v] for v in a) + "}"    # noqa: E731
    if c["kind"] == "trivial":
        return "info", "Es gibt keinen Pfeil von X nach Y: Die Wirkung ist in jeder Gruppe null und trivial übertragbar.", []
    if c["kind"] == "direkt":
        return "success", "Die Wirkung ist **direkt übertragbar**: Gegeben X ist Y von S getrennt (im Diagramm ohne Pfeile in X); die Wirkung der Quelle gilt im Ziel unverändert.", []
    if c["kind"] == "menge":
        small = min((a for a in sets if a), key=len)
        return "success", (f"Die Wirkung ist **nur über eine Menge übertragbar**: S-zulässig ist {', '.join(names(a) for a in sets if a)}. Die Wirkung der Quelle muss über die Verteilung von {names(small)} im Ziel "
                           "neu gemittelt werden (Transportformel); ohne diese Messung im Ziel nicht."), Sel.open_paths(parents, ())
    return "error", "Die Wirkung ist **nicht übertragbar**: Für keine Menge aus Z und W ist Y von S getrennt. Es braucht Daten aus dem Ziel.", Sel.open_paths(parents, (Sel.Z, Sel.W))
