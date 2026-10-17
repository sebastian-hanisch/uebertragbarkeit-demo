"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35
Z_CRIT = 1.959964

# Welt: Y = BASE + y_shift + BZ·Z + τ(Z)·X + SIGMA·ε, τ(z) = TAU0 + τ1·z + τ2·z² (+ γ im Ziel); X = 1[CONFOUND·(Z − Mitte) + η > 0]; Quelle Z ~ N(0, 1), Ziel Z ~ N(Δ, 1)
BASE, TAU0, BZ, SIGMA, CONFOUND = 50.0, 2.0, 2.0, 3.0, 0.5
NS_MIN, NS_MAX, NS_STEP, DEFAULT_NS = 100, 2000, 100, 500          # Standorte mit Messungen in der Quelle
NT_MIN, NT_MAX, NT_STEP, DEFAULT_NT = 100, 2000, 100, 500          # Standorte der Zielgruppe (nur Z gemessen)
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.0, 3.0, 0.25, 1.0
TAU1_MIN, TAU1_MAX, TAU1_STEP, DEFAULT_TAU1 = 0.0, 3.0, 0.25, 1.5
TAU2_MIN, TAU2_MAX, TAU2_STEP, DEFAULT_TAU2 = 0.0, 1.0, 0.25, 0.0
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.0, 2.0, 0.25, 0.0
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = 0.0, 20.0, 5.0, 0.0
MODELS = ("linear", "quadratisch")
DEFAULT_MODEL = MODELS[0]

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1020000
EXP_N, EXP_NT, EXP_REPS = 500, 500, 300
EXP_DELTAS, EXP_TAU1S = (0.0, 0.5, 1.0, 2.0), (0.0, 0.75, 1.5, 3.0)            # Experiment 1: Verschiebung × Modifikator
EXP_SHIFTS, EXP_GAMMAS, EXP_PILOTS = (0.0, 5.0, 20.0), (0.0, 0.5, 1.0, 2.0), (60, 200, 600)    # Experiment 2
EXP_OVERLAP_DELTAS, EXP_TAU2, EXP_ESS_DRAWS = (0.5, 1.0, 2.0, 3.0), 0.5, 200000   # Experiment 3
EXP_TABLES = 6                                                                  # Experiment 4: Zufallstabellen je Diagramm und Menge

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(delta=DEFAULT_DELTA, tau1=DEFAULT_TAU1, tau2=DEFAULT_TAU2, gamma=DEFAULT_GAMMA, shift=DEFAULT_SHIFT, ns=DEFAULT_NS, nt=DEFAULT_NT, seed=DEFAULT_SEED, model=DEFAULT_MODEL)
    base.update(kw)
    return base


PRESETS = {
    "Verschobene Verteilung (Voreinstellung)": preset(),
    "Keine Verschiebung": preset(delta=0.0),
    "Kein Effektmodifikator": preset(tau1=0.0),
    "Zusatzwirkung im Ziel": preset(gamma=1.0),
    "Niveauunterschied": preset(shift=20.0),
    "Geringe Überlappung": preset(delta=2.5, tau2=0.5),
}
PRESET_HELP = {
    "Verschobene Verteilung (Voreinstellung)": "Im Ziel schwankt die Nachfrage stärker (Z im Mittel 1,0 höher), und der Faktor wirkt dort stärker. Die naive Übertragung liegt daneben, die Umgewichtung nicht.",
    "Keine Verschiebung": "Quelle und Ziel gleichen sich in Z: Es gibt nichts zu übertragen, die naive Übertragung trifft.",
    "Kein Effektmodifikator": "Die Wirkung hängt nicht von Z ab (τ₁ = 0): Auch bei verschobener Verteilung trifft die naive Übertragung.",
    "Zusatzwirkung im Ziel": "Im Ziel wirkt zusätzlich etwas, das in der Quelle fehlt und nicht gemessen ist (γ = 1): Keine Rechnung mit Z kann das ausgleichen, die Übertragung liegt um γ daneben.",
    "Niveauunterschied": "Der mittlere Servicegrad im Ziel liegt um 20 höher, die Wirkung des Faktors bleibt gleich: Die Wirkung überträgt sich, das Niveau nicht.",
    "Geringe Überlappung": "Das Ziel liegt weit in den Rändern der Quelle (Δ = 2,5) und der Modifikator ist gekrümmt: Das lineare Modell extrapoliert falsch, die Gewichte sind sehr ungleich.",
}
