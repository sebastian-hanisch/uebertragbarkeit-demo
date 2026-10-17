"""Selektionsdiagramme (Pearl/Bareinboim) für vier beobachtete Größen Z, W, X, Y und den Selektionsknoten S, der auf die Größen zeigt, in denen sich Quelle und Ziel unterscheiden.
Dazu die S-Zulässigkeit (über dsep_graph), die Begründung durch offene Pfade und die **exakte** Gegenprobe: die Transportformel gegen die wahre Zielwirkung bei binären Größen.
Nur Standardbibliothek und dsep_graph (wortgleich aus den Vorgängern), kein Streamlit."""

from __future__ import annotations

import itertools

import dsep_graph as G
from kausal_generator import SplitMix64

S, Z, W, X, Y = 0, 1, 2, 3, 4
NAMES = ("S", "Z", "W", "X", "Y")
OBSERVED = (Z, W, X, Y)
EDGES = ((Z, W), (Z, X), (Z, Y), (W, X), (W, Y), (X, Y))            # Bit k einer Kantenmaske gehört zu EDGES[k]
SETS = ((), (Z,), (W,), (Z, W))                                      # mögliche S-zulässige Mengen (ohne X und Y)


def edges_of(mask: int) -> tuple:
    return tuple(e for k, e in enumerate(EDGES) if mask >> k & 1)


def s_targets_of(mask: int) -> tuple:
    return tuple(v for k, v in enumerate(OBSERVED) if mask >> k & 1)


def parents_of(edge_mask: int, s_mask: int) -> tuple:
    """Eltern-Mengen aller fünf Knoten (S = 0 ohne Eltern); S zeigt auf die Größen der S-Maske."""
    edges, targets = edges_of(edge_mask), s_targets_of(s_mask)
    out = [()] * 5
    for v in OBSERVED:
        ps = [a for a, b in edges if b == v] + ([S] if v in targets else [])
        out[v] = tuple(sorted(ps))
    return tuple(out)


def has_path(parents: tuple, x: int, y: int) -> bool:
    """Gerichteter Pfad x → … → y?"""
    todo, seen = [y], {y}
    while todo:
        v = todo.pop()
        for a in parents[v]:
            if a == x:
                return True
            if a not in seen:
                seen.add(a)
                todo.append(a)
    return False


def cut_into_x(parents: tuple) -> tuple:
    """Das Diagramm ohne alle Pfeile in X (G ohne eingehende Pfeile von X)."""
    return tuple(() if v == X else ps for v, ps in enumerate(parents))


def s_admissible(parents: tuple, adj=()) -> bool:
    """S-zulässig: (S ⟂ Y | X, adj) im Diagramm ohne Pfeile in X. Mit adj = () ist die Wirkung direkt übertragbar."""
    return G.d_separated(cut_into_x(parents), S, Y, (X, *adj))


def classify(parents: tuple) -> dict:
    """Urteil für ein Selektionsdiagramm: "trivial" (kein Pfad X → Y, die Wirkung ist in jeder Gruppe null), "direkt", "menge" (nur über eine zulässige Menge) oder "keine"."""
    ok = {a: s_admissible(parents, a) for a in SETS}
    if not has_path(parents, X, Y):
        kind = "trivial"
    elif ok[()]:
        kind = "direkt"
    elif any(ok.values()):
        kind = "menge"
    else:
        kind = "keine"
    return {"kind": kind, "admissible": tuple(a for a in SETS if ok[a])}


def open_paths(parents: tuple, adj=()) -> list:
    """Offene Pfade zwischen S und Y gegeben X und adj im Diagramm ohne Pfeile in X: die Begründung, warum ein Diagramm nicht übertragbar ist."""
    return [p for p in G.explain_paths(cut_into_x(parents), S, Y, (X, *adj)) if not p.blocked]


def path_text(path) -> str:
    out = NAMES[path.nodes[0]]
    for arrow, node in zip(path.arrows, path.nodes[1:]):
        out += f" {arrow} {NAMES[node]}"
    return out


# ----------------------------------------------------------------------------- exakte Gegenprobe (binäre Größen)

def random_tables(parents: tuple, rng: SplitMix64) -> dict:
    """Zufällige Tabellen P(V = 1 | Eltern aus Z, W, X) für jede beobachtete Größe (S ist kein Elter in der Tabelle)."""
    out = {}
    for v in OBSERVED:
        k = len([a for a in parents[v] if a != S])
        out[v] = 0.1 + 0.8 * rng.uniform(2 ** k)
    return out


def _joint(parents: tuple, tables: dict, do_x=None) -> dict:
    out = {}
    for z, w, x, y in itertools.product((0, 1), repeat=4):
        if do_x is not None and x != do_x:
            continue
        state = {Z: z, W: w, X: x, Y: y}
        p = 1.0
        for v in OBSERVED:
            if v == X and do_x is not None:
                continue
            idx = 0
            for a in parents[v]:
                if a != S:
                    idx = idx * 2 + state[a]
            q = tables[v][idx]
            p *= q if state[v] == 1 else 1.0 - q
        out[(z, w, x, y)] = p
    return out


def target_effect(parents: tuple, target_tables: dict) -> float:
    """Wahre Wirkung im Ziel: P(Y = 1 | do(X = 1)) − P(Y = 1 | do(X = 0)) nach der abgeschnittenen Faktorisierung."""
    return sum(p for (_, _, _, y), p in _joint(parents, target_tables, 1).items() if y == 1) - sum(p for (_, _, _, y), p in _joint(parents, target_tables, 0).items() if y == 1)


def transport_formula(parents: tuple, source_tables: dict, target_tables: dict, adj=()) -> float:
    """Σ_a P_Quelle(y | do(x), a) · P_Ziel(a) für die Menge adj ⊆ {Z, W} (leer: die Wirkung der Quelle unverändert)."""
    total = []
    target = _joint(parents, target_tables)
    for do in (1, 0):
        dist = _joint(parents, source_tables, do)
        value = 0.0
        for a in itertools.product((0, 1), repeat=len(adj)):
            def match(state):
                return all({Z: state[0], W: state[1]}[v] == s for v, s in zip(adj, a))

            p_a_do = sum(p for st, p in dist.items() if match(st))
            p_ya_do = sum(p for st, p in dist.items() if st[3] == 1 and match(st))
            p_a_tgt = sum(p for st, p in target.items() if match(st))
            if p_a_do > 0:
                value += p_ya_do / p_a_do * p_a_tgt
        total.append(value)
    return total[0] - total[1]


def tables_pair(parents: tuple, rng: SplitMix64) -> tuple[dict, dict]:
    """Quelle und Ziel: In den Größen, auf die S zeigt, hat das Ziel eine eigene Tabelle, sonst dieselbe."""
    source = random_tables(parents, rng)
    fresh = random_tables(parents, rng)
    return source, {v: (fresh[v] if S in parents[v] else source[v]) for v in OBSERVED}
