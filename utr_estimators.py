"""Schätzer: Wirkungsfunktion der Quelle (Regression mit Wechselwirkung), naive Übertragung, transportierte Wirkung (Zielstichprobe von Z), gewichtete Wirkung (bekannte Dichten), effektive
Stichprobe und die Pilotstudie. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import utr_constants as C
import utr_world as W
from kausal_generator import SplitMix64


def ols(y: np.ndarray, columns) -> tuple[np.ndarray, np.ndarray]:
    """Kleinste Quadrate mit Achsenabschnitt: (Koeffizienten, Kovarianzmatrix der Koeffizienten)."""
    d = np.column_stack([np.ones(len(y))] + list(columns))
    beta = np.linalg.solve(d.T @ d, d.T @ y)
    res = y - d @ beta
    return beta, np.linalg.inv(d.T @ d) * (res @ res) / (len(y) - d.shape[1])


@dataclass(frozen=True)
class EffectFit:
    """Wirkungsfunktion τ̂(z) = Σ_k b_k z^k, k = 0 .. degree (Koeffizienten von x, x z, x z² in y ~ x + z + x z (+ z² + x z²))."""
    degree: int
    coef: np.ndarray          # b_0 .. b_degree
    cov: np.ndarray           # Kovarianz dieser Koeffizienten

    def at(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return sum(self.coef[k] * z ** k for k in range(self.degree + 1))


def fit_effect_function(z: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int = 1) -> EffectFit:
    cols = [x, z, x * z] if degree == 1 else [x, z, x * z, z * z, x * z * z]
    beta, cov = ols(y, cols)
    idx = [1, 3] if degree == 1 else [1, 3, 5]                                    # x, x z (, x z²) in der Reihenfolge der Spalten
    return EffectFit(degree, beta[idx], cov[np.ix_(idx, idx)])


def _moments(z_sample: np.ndarray, weights: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Gewichtete Momente E[z^k], k = 1 .. degree, und die Kovarianz dieser Schätzungen (Näherung über die Einflusswerte)."""
    feats = np.column_stack([z_sample ** k for k in range(1, degree + 1)])
    w = weights / weights.sum()
    m = w @ feats
    centered = feats - m
    cov = (centered * (w ** 2)[:, None]).T @ centered
    return m, cov


def evaluate(fit: EffectFit, z_sample: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Mittlere Wirkung Σ_k b_k E[z^k] mit den (gewichteten) Momenten der Stichprobe `z_sample` und dem Delta-Standardfehler aus Koeffizienten und Momenten."""
    w = np.ones(len(z_sample)) if weights is None else weights
    m, cov_m = _moments(z_sample, w, fit.degree)
    g = np.concatenate([[1.0], m])
    est = float(g @ fit.coef)
    var = float(g @ fit.cov @ g + fit.coef[1:] @ cov_m @ fit.coef[1:])
    se = float(np.sqrt(max(var, 0.0)))
    return {"estimate": est, "se": se, "lo": est - C.Z_CRIT * se, "hi": est + C.Z_CRIT * se}


def naive_transfer(fit: EffectFit, z_source: np.ndarray) -> dict:
    """Die mittlere Wirkung in der Quelle (an den Z der Quelle ausgewertet), unverändert für das Ziel übernommen."""
    return evaluate(fit, z_source)


def transported(fit: EffectFit, z_target: np.ndarray) -> dict:
    """Die Wirkungsfunktion der Quelle, über die gemessene Verteilung von Z im Ziel gemittelt."""
    return evaluate(fit, z_target)


def weighted(fit: EffectFit, z_source: np.ndarray, delta: float) -> dict:
    """Die Wirkungsfunktion der Quelle, über die Quellstichprobe mit den Gewichten p_Ziel/p_Quelle (bekannte Dichten) gemittelt."""
    return evaluate(fit, z_source, W.gauss_weights(z_source, delta))


def ess(weights: np.ndarray) -> float:
    """Effektive Stichprobe (Σw)² / Σw²."""
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def pilot_contradicts(world: W.World, n_pilot: int, rng: SplitMix64, transferred: dict) -> bool:
    """Eine kleine randomisierte Studie im Ziel: Widerspricht die Differenz der Mittelwerte der übertragenen Wirkung (95 %, Streuung beider Seiten)?"""
    s = W.simulate(world, n_pilot, rng, target=True, randomized=True)
    x, y = s["x"][0], s["y"][0]
    d = y[x == 1].mean() - y[x == 0].mean()
    sd = np.sqrt(y[x == 1].var(ddof=1) / max((x == 1).sum(), 2) + y[x == 0].var(ddof=1) / max((x == 0).sum(), 2))
    return bool(abs(d - transferred["estimate"]) > C.Z_CRIT * np.sqrt(sd ** 2 + transferred["se"] ** 2))
