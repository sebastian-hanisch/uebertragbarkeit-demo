"""Vehikel: Quelle und Ziel, zwei Gruppen von Standorten. Z (Nachfrageschwankung) ist ein Effektmodifikator: Die Wirkung des hohen Faktors X auf den Servicegrad Y ist τ(z) = τ0 + τ1 z + τ2 z².
In der Quelle ist Z ~ N(0, 1), im Ziel Z ~ N(Δ, 1); im Ziel kann außerdem ein unbeobachteter Zusatzeffekt γ und ein Niveauunterschied wirken. Nur numpy und der Generator der Linie, kein Streamlit.

Die Zuteilung ist beobachtet: X = 1[CONFOUND (Z − Mitte) + η > 0]; in der Pilotstudie (randomisiert) ist X ein Münzwurf."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import utr_constants as C
from kausal_generator import SplitMix64


@dataclass(frozen=True)
class World:
    delta: float = C.DEFAULT_DELTA
    tau1: float = C.DEFAULT_TAU1
    tau2: float = C.DEFAULT_TAU2
    gamma: float = C.DEFAULT_GAMMA
    shift: float = C.DEFAULT_SHIFT


def effect_function(w: World, z, target: bool = False):
    """Wirkung des Faktors an der Stelle z: τ0 + τ1 z + τ2 z² (+ γ im Ziel)."""
    z = np.asarray(z, dtype=float)
    return C.TAU0 + w.tau1 * z + w.tau2 * z ** 2 + (w.gamma if target else 0.0)


def source_truth(w: World) -> float:
    """Mittlere Wirkung in der Quelle: E[τ(Z)] mit Z ~ N(0, 1): τ0 + τ2."""
    return C.TAU0 + w.tau2


def target_truth(w: World) -> float:
    """Mittlere Wirkung im Ziel: E[τ(Z)] mit Z ~ N(Δ, 1): τ0 + τ1 Δ + τ2 (Δ² + 1) + γ."""
    return C.TAU0 + w.tau1 * w.delta + w.tau2 * (w.delta ** 2 + 1.0) + w.gamma


def simulate(w: World, n: int, rng: SplitMix64, target: bool = False, reps: int = 1, randomized: bool = False) -> dict:
    """z, x, y mit Form (reps, n) für die Quelle oder das Ziel. `randomized`: x ist ein Münzwurf (Pilotstudie), sonst folgt x der Zuteilungsregel."""
    mean = w.delta if target else 0.0
    draw = rng.normal(reps * n * 3).reshape(reps, n, 3)
    z = mean + draw[..., 0]
    if randomized:
        x = (rng.uniform(reps * n).reshape(reps, n) < 0.5).astype(float)
    else:
        x = (C.CONFOUND * (z - mean) + draw[..., 1] > 0).astype(float)
    level = C.BASE + (w.shift if target else 0.0)
    y = level + C.BZ * z + effect_function(w, z, target) * x + C.SIGMA * draw[..., 2]
    return {"z": z, "x": x, "y": y}


def gauss_weights(z: np.ndarray, delta: float) -> np.ndarray:
    """Gewichte p_Ziel(z) / p_Quelle(z) für N(Δ, 1) gegen N(0, 1): exp(Δ z − Δ²/2)."""
    return np.exp(delta * z - delta * delta / 2.0)


def exact_ess_ratio(delta: float) -> float:
    """Effektive Stichprobe der Gewichte als Anteil von n: 1 / E_Quelle[w²] = exp(−Δ²)."""
    return float(np.exp(-delta * delta))
