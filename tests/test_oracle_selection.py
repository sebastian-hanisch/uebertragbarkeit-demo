"""Orakel: Die Transportformel gegen die wahre Zielwirkung bei binären Größen, exakt, für alle 1 024 Selektionsdiagramme. S-Zulässigkeit ist hinreichend und (für diese Klasse, generisch) notwendig."""

import itertools

import pytest

import utr_selection as Sel
from kausal_generator import SplitMix64


def _all_diagrams():
    for em, sm in itertools.product(range(2 ** len(Sel.EDGES)), range(2 ** len(Sel.OBSERVED))):
        yield em, sm, Sel.parents_of(em, sm)


def test_counts_of_the_kinds():
    kinds = {}
    for _, _, p in _all_diagrams():
        k = Sel.classify(p)["kind"]
        kinds[k] = kinds.get(k, 0) + 1
    assert kinds == {"trivial": 512, "direkt": 136, "menge": 120, "keine": 256}


def test_trivial_diagrams_are_exactly_those_without_an_arrow_from_x_to_y():
    xy = 1 << Sel.EDGES.index((Sel.X, Sel.Y))
    for em, sm, p in _all_diagrams():
        assert (Sel.classify(p)["kind"] == "trivial") == (not em & xy)


def test_the_criterion_is_sufficient_and_necessary_on_exact_tables():
    rng = SplitMix64(11)
    right = wrong = 0
    for em, sm, p in _all_diagrams():
        c = Sel.classify(p)
        if c["kind"] == "trivial":
            continue
        for _ in range(2):
            source, target = Sel.tables_pair(p, rng)
            truth = Sel.target_effect(p, target)
            for adj in Sel.SETS:
                equal = abs(Sel.transport_formula(p, source, target, adj) - truth) < 1e-9
                if adj in c["admissible"]:
                    assert equal, (em, sm, adj)
                    right += 1
                else:
                    assert not equal, (em, sm, adj)
                    wrong += 1
    assert right > 1500 and wrong > 2500


def test_without_s_arrows_every_diagram_is_directly_transportable():
    for em in range(64):
        p = Sel.parents_of(em, 0)
        assert Sel.s_admissible(p, ()) and Sel.classify(p)["kind"] in ("trivial", "direkt")


def test_s_pointing_into_y_is_never_transportable():
    for em in range(64):
        p = Sel.parents_of(em, 0b1000)
        assert Sel.classify(p)["kind"] in ("trivial", "keine")


def test_admissible_sets_never_contain_x_or_y():
    for em, sm, p in _all_diagrams():
        for a in Sel.classify(p)["admissible"]:
            assert Sel.X not in a and Sel.Y not in a
