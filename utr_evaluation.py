"""Auswertung: Quelle und Ziel mit allen Schätzern, die vier Experimente über feste Ströme und die Verdikte. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import utr_constants as C
import utr_estimators as E
import utr_selection as Sel
import utr_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


@dataclass(frozen=True)
class Settings:
    delta: float = C.DEFAULT_DELTA
    tau1: float = C.DEFAULT_TAU1
    tau2: float = C.DEFAULT_TAU2
    gamma: float = C.DEFAULT_GAMMA
    shift: float = C.DEFAULT_SHIFT
    ns: int = C.DEFAULT_NS
    nt: int = C.DEFAULT_NT
    seed: int = C.DEFAULT_SEED
    model: str = C.DEFAULT_MODEL

    @property
    def world(self) -> W.World:
        return W.World(self.delta, self.tau1, self.tau2, self.gamma, self.shift)

    @property
    def degree(self) -> int:
        return 1 if self.model == C.MODELS[0] else 2


ROW_NAMES = ("Naive Übertragung", "Transportiert (Z im Ziel gemessen)", "Gewichtet (bekannte Dichten)")


@dataclass
class Row:
    name: str
    estimate: float
    se: float
    lo: float
    hi: float
    error: float
    covers: bool


@dataclass
class Analysis:
    settings: Settings
    truth: float            # mittlere Wirkung im Ziel
    source_truth: float     # mittlere Wirkung in der Quelle
    rows: list
    fit: E.EffectFit
    ess_ratio: float        # effektive Stichprobe der Gewichte als Anteil von n (Stichprobe)
    ess_exact: float        # exp(−Δ²)
    source_mean_z: float
    target_mean_z: float
    level_difference: float  # Unterschied des mittleren Servicegrads (Ziel − Quelle), aus den Daten der Quelle und einer Zielstichprobe


def _row(name: str, r: dict, truth: float) -> Row:
    return Row(name, r["estimate"], r["se"], r["lo"], r["hi"], r["estimate"] - truth, bool(r["lo"] <= truth <= r["hi"]))


def analyse(s: Settings) -> Analysis:
    w = s.world
    src = W.simulate(w, s.ns, SplitMix64(s.seed))
    tgt = W.simulate(w, s.nt, SplitMix64(s.seed + 1), target=True)
    z, x, y = src["z"][0], src["x"][0], src["y"][0]
    fit = E.fit_effect_function(z, x, y, s.degree)
    truth = W.target_truth(w)
    rows = [_row(ROW_NAMES[0], E.naive_transfer(fit, z), truth), _row(ROW_NAMES[1], E.transported(fit, tgt["z"][0]), truth), _row(ROW_NAMES[2], E.weighted(fit, z, s.delta), truth)]
    weights = W.gauss_weights(z, s.delta)
    return Analysis(s, truth, W.source_truth(w), rows, fit, E.ess(weights) / len(z), W.exact_ess_ratio(s.delta), float(z.mean()), float(tgt["z"][0].mean()), float(tgt["y"][0].mean() - y.mean()))


# ----------------------------------------------------------------------------- Experiment 1: Verschiebung × Modifikator

def shift_experiment(deltas=C.EXP_DELTAS, tau1s=C.EXP_TAU1S, n: int = C.EXP_N, nt: int = C.EXP_NT, reps: int = C.EXP_REPS) -> list[dict]:
    """Je Zelle (Δ, τ₁): naive Übertragung, transportierte und gewichtete Wirkung über feste Wiederholungen gegen die Wahrheit im Ziel (die naive Übertragung ist zugleich der Fall „Z nur in der Quelle gemessen“)."""
    rows = []
    for i, delta in enumerate(deltas):
        for j, tau1 in enumerate(tau1s):
            w = W.World(delta, tau1)
            truth = W.target_truth(w)
            seed = C.SEED_BASE + 10 * i + j
            src = W.simulate(w, n, SplitMix64(seed), reps=reps)
            tgt = W.simulate(w, nt, SplitMix64(seed + 500), target=True, reps=reps)
            naive, trans, wtd, cover = [], [], [], []
            for r in range(reps):
                fit = E.fit_effect_function(src["z"][r], src["x"][r], src["y"][r])
                naive.append(E.naive_transfer(fit, src["z"][r])["estimate"])
                t = E.transported(fit, tgt["z"][r])
                trans.append(t["estimate"])
                cover.append(t["lo"] <= truth <= t["hi"])
                wtd.append(E.weighted(fit, src["z"][r], delta)["estimate"])
            rows.append({"delta": delta, "tau1": tau1, "truth": truth, "naive": float(np.mean(naive)), "naive_bias": float(np.mean(naive) - truth), "transported": float(np.mean(trans)),
                         "transported_bias": float(np.mean(trans) - truth), "transported_sd": float(np.std(trans)), "cover": float(np.mean(cover)), "weighted_bias": float(np.mean(wtd) - truth),
                         "weighted_sd": float(np.std(wtd))})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: anderer Mechanismus und Pilotstudie

def mechanism_experiment(n: int = C.EXP_N, reps: int = C.EXP_REPS) -> dict:
    """(a) Niveauunterschied im Ziel: Die Wirkung überträgt sich, das Niveau nicht. (b) Zusatzwirkung γ im Ziel (unbeobachtet): Die Übertragung liegt um γ daneben. (c) Eine kleine randomisierte Zielstudie
    widerspricht der Übertragung nur selten (Anteil der Läufe)."""
    level = []
    for i, shift in enumerate(C.EXP_SHIFTS):
        w = W.World(0.0, 1.5, shift=shift)
        src = W.simulate(w, n, SplitMix64(C.SEED_BASE + 100 + i), reps=reps)
        tgt = W.simulate(w, n, SplitMix64(C.SEED_BASE + 120 + i), target=True, reps=reps)
        est, diff = [], []
        for r in range(reps):
            fit = E.fit_effect_function(src["z"][r], src["x"][r], src["y"][r])
            est.append(E.naive_transfer(fit, src["z"][r])["estimate"])
            diff.append(tgt["y"][r].mean() - src["y"][r].mean())
        level.append({"shift": shift, "truth": W.target_truth(w), "transferred": float(np.mean(est)), "level_difference": float(np.mean(diff))})
    gammas = []
    for i, gamma in enumerate(C.EXP_GAMMAS):
        w = W.World(0.0, 1.5, gamma=gamma)
        truth = W.target_truth(w)
        src = W.simulate(w, n, SplitMix64(C.SEED_BASE + 140 + i), reps=reps)
        rng = SplitMix64(C.SEED_BASE + 160 + i)
        err, cover = [], []
        contradict = {p: [] for p in C.EXP_PILOTS}
        for r in range(reps):
            fit = E.fit_effect_function(src["z"][r], src["x"][r], src["y"][r])
            t = E.naive_transfer(fit, src["z"][r])
            err.append(t["estimate"] - truth)
            cover.append(t["lo"] <= truth <= t["hi"])
            for p in C.EXP_PILOTS:
                contradict[p].append(E.pilot_contradicts(w, p, rng, t))
        gammas.append({"gamma": gamma, "truth": truth, "error": float(np.mean(err)), "cover": float(np.mean(cover)), "pilot": {p: float(np.mean(v)) for p, v in contradict.items()}})
    return {"level": level, "gammas": gammas, "n": n, "reps": reps}


# ----------------------------------------------------------------------------- Experiment 3: Überlappung und Extrapolation

def overlap_experiment(n: int = C.EXP_N, reps: int = C.EXP_REPS) -> list[dict]:
    """Je Verschiebung Δ: die effektive Stichprobe der Gewichte (exakt exp(−Δ²) und aus einer großen Stichprobe) und der Preis der Extrapolation bei quadratischem Modifikator: lineares Modell (Verzerrung)
    gegen quadratisches (Streuung)."""
    rows = []
    rng = SplitMix64(C.SEED_BASE + 200)
    big = rng.normal(C.EXP_ESS_DRAWS)
    for i, delta in enumerate(C.EXP_OVERLAP_DELTAS):
        w = W.World(delta, 1.0, C.EXP_TAU2)
        truth = W.target_truth(w)
        sim = W.simulate(w, n, SplitMix64(C.SEED_BASE + 210 + i), reps=reps)
        tgt = W.simulate(w, C.EXP_NT, SplitMix64(C.SEED_BASE + 230 + i), target=True, reps=reps)
        lin, quad, lin_w = [], [], []
        for r in range(reps):
            lin.append(E.transported(E.fit_effect_function(sim["z"][r], sim["x"][r], sim["y"][r], 1), tgt["z"][r])["estimate"])
            quad.append(E.transported(E.fit_effect_function(sim["z"][r], sim["x"][r], sim["y"][r], 2), tgt["z"][r])["estimate"])
            lin_w.append(E.weighted(E.fit_effect_function(sim["z"][r], sim["x"][r], sim["y"][r], 2), sim["z"][r], delta)["estimate"])
        rows.append({"delta": delta, "truth": truth, "ess_exact": W.exact_ess_ratio(delta), "ess_sample": E.ess(W.gauss_weights(big, delta)) / len(big), "linear_bias": float(np.mean(lin) - truth),
                     "linear_sd": float(np.std(lin)), "quadratic_bias": float(np.mean(quad) - truth), "quadratic_sd": float(np.std(quad)), "weighted_sd": float(np.std(lin_w))})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: alle Selektionsdiagramme (exakt)

def selection_experiment(tables: int = C.EXP_TABLES) -> dict:
    """Alle 2⁶ Kantenmengen × 2⁴ S-Pfeile für vier binäre Größen: Anteile direkt / über Menge / keine (nur mit Pfeil X → Y) und die Gegenprobe der Transportformel gegen die wahre Zielwirkung:
    Ist eine Menge S-zulässig, ist die Formel gleich der Wahrheit; ist sie es nicht, ist sie ungleich."""
    rng = SplitMix64(C.SEED_BASE + 300)
    kinds = {"trivial": 0, "direkt": 0, "menge": 0, "keine": 0}
    by_s: dict = {}
    wrong_true = wrong_false = checked_true = checked_false = 0
    for em in range(2 ** len(Sel.EDGES)):
        for sm in range(2 ** len(Sel.OBSERVED)):
            parents = Sel.parents_of(em, sm)
            c = Sel.classify(parents)
            kinds[c["kind"]] += 1
            if c["kind"] == "trivial":
                continue
            row = by_s.setdefault(bin(sm).count("1"), {"diagrams": 0, "direkt": 0, "menge": 0, "keine": 0})
            row["diagrams"] += 1
            row[c["kind"]] += 1
            for _ in range(tables):
                source, target = Sel.tables_pair(parents, rng)
                truth = Sel.target_effect(parents, target)
                for adj in Sel.SETS:
                    equal = abs(Sel.transport_formula(parents, source, target, adj) - truth) < 1e-9
                    if adj in c["admissible"]:
                        checked_true += 1
                        wrong_true += not equal
                    else:
                        checked_false += 1
                        wrong_false += equal
    return {"diagrams": sum(kinds.values()), "kinds": kinds, "by_s": dict(sorted(by_s.items())), "checked_true": checked_true, "wrong_true": int(wrong_true), "checked_false": checked_false,
            "equal_although_false": int(wrong_false), "tables": tables}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    by = {r.name: r for r in a.rows}
    n, t, wt = by[ROW_NAMES[0]], by[ROW_NAMES[1]], by[ROW_NAMES[2]]
    s = a.settings
    text = (f"Die Wahrheit im Ziel ist {de(a.truth)} (in der Quelle {de(a.source_truth)}). Die naive Übertragung schätzt {de(n.estimate)} (Fehler {signed(n.error)}), die transportierte Wirkung {de(t.estimate)} "
            f"(Fehler {signed(t.error)}, das Intervall {'deckt' if t.covers else 'verfehlt'} die Wahrheit), die gewichtete {de(wt.estimate)} bei einer effektiven Stichprobe von {pct(a.ess_ratio, 1)}.")
    if s.gamma > 0:
        return "warning", text + f" Im Ziel wirkt eine unbeobachtete Zusatzwirkung von {de(s.gamma)}: Sie lässt sich aus Z nicht ausgleichen, die Übertragung liegt um sie daneben."
    if not t.covers:
        return "warning", text + " Das Intervall der transportierten Wirkung verfehlt die Wahrheit: Das Wirkungsmodell der Quelle passt hier nicht zur Form des Modifikators."
    return "success", text


def verdict_shift(rows: list) -> tuple[str, str]:
    by = {(r["delta"], r["tau1"]): r for r in rows}
    a, b = by[(1.0, 1.5)], by[(2.0, 3.0)]
    cover = [r["cover"] for r in rows]
    return "info", (f"Bei Δ = 1 und τ₁ = 1,5 liegt die naive Übertragung im Mittel um {signed(a['naive_bias'])} daneben, die transportierte um {signed(a['transported_bias'])}; bei Δ = 2 und τ₁ = 3 sind es {signed(b['naive_bias'])} "
                    f"gegen {signed(b['transported_bias'])}. Die Deckung der transportierten Wirkung liegt in allen {len(rows)} Zellen zwischen {pct(min(cover))} und {pct(max(cover))}; die Streuung wächst von "
                    f"{de(by[(0.0, 0.0)]['transported_sd'])} (Δ = 0) auf {de(b['transported_sd'])} (Δ = 2).")


def verdict_mechanism(res: dict) -> tuple[str, str]:
    lv, g = res["level"][-1], {r["gamma"]: r for r in res["gammas"]}
    last = max(g)
    return "info", (f"Liegt der Servicegrad im Ziel um {de(lv['level_difference'], 1)} höher, bleibt die übertragene Wirkung {de(lv['transferred'])} (Wahrheit {de(lv['truth'])}): Die Wirkung überträgt sich, das Niveau nicht. "
                    f"Eine Zusatzwirkung von {de(last)} im Ziel macht den Fehler {signed(g[last]['error'])}, die Deckung fällt auf {pct(g[last]['cover'])}; eine Pilotstudie mit {min(g[last]['pilot'])} Standorten "
                    f"widerspricht der Übertragung nur in {pct(g[last]['pilot'][min(g[last]['pilot'])])} der Läufe, mit {max(g[last]['pilot'])} in {pct(g[last]['pilot'][max(g[last]['pilot'])])}.")


def verdict_overlap(rows: list) -> tuple[str, str]:
    by = {r["delta"]: r for r in rows}
    a, b = by[1.0], by[2.0]
    return "info", (f"Die effektive Stichprobe der Gewichte ist bei Δ = 1 nur {pct(a['ess_exact'], 0)} von n, bei Δ = 2 {pct(b['ess_exact'], 1)}. Bei gekrümmtem Modifikator liegt das lineare Modell bei Δ = 2 um {signed(b['linear_bias'])} daneben, "
                    f"das quadratische um {signed(b['quadratic_bias'])}, streut dafür mit {de(b['quadratic_sd'])} gegen {de(b['linear_sd'])}.")


def verdict_selection(res: dict) -> tuple[str, str]:
    k = res["kinds"]
    nontrivial = res["diagrams"] - k["trivial"]
    return "info", (f"Von {res['diagrams']} Selektionsdiagrammen haben {k['trivial']} keinen Pfeil von X nach Y (die Wirkung ist überall null). Von den übrigen {nontrivial} ist die Wirkung in {k['direkt']} ({pct(k['direkt'] / nontrivial, 1)}) "
                    f"direkt übertragbar, in {k['menge']} ({pct(k['menge'] / nontrivial, 1)}) nur über eine zulässige Menge und in {k['keine']} ({pct(k['keine'] / nontrivial, 1)}) nicht. Ist das Kriterium erfüllt, weicht die Formel in "
                    f"{res['wrong_true']} von {res['checked_true']} Fällen von der wahren Zielwirkung ab; ist es nicht erfüllt, trifft sie in {res['equal_although_false']} von {res['checked_false']} Fällen trotzdem.")
