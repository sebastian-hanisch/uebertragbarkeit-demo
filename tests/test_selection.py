"""Selektionsdiagramme: Masken, Eltern-Mengen, S-Zulässigkeit, Urteil und die exakte binäre Rechnung, an Hand-Instanzen."""

import itertools

import pytest

import utr_selection as Sel
from kausal_generator import SplitMix64

CHAIN = (Sel.EDGES.index((Sel.Z, Sel.X)), Sel.EDGES.index((Sel.Z, Sel.Y)), Sel.EDGES.index((Sel.X, Sel.Y)))
CHAIN_MASK = sum(1 << k for k in CHAIN)


def _s_mask(*targets):
    return sum(1 << Sel.OBSERVED.index(v) for v in targets)


def test_masks_by_hand():
    assert Sel.edges_of(0) == () and Sel.edges_of(1) == ((Sel.Z, Sel.W),) and Sel.edges_of(CHAIN_MASK) == ((Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)) and len(Sel.edges_of(63)) == 6
    assert Sel.s_targets_of(0) == () and Sel.s_targets_of(0b0101) == (Sel.Z, Sel.X) and Sel.s_targets_of(15) == Sel.OBSERVED


def test_parents_by_hand():
    p = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z))
    assert p == ((), (0,), (), (1,), (1, 3)) and Sel.parents_of(CHAIN_MASK, 0) == ((), (), (), (1,), (1, 3))
    q = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Y, Sel.X))
    assert q[Sel.X] == (Sel.S, Sel.Z) and q[Sel.Y] == (Sel.S, Sel.Z, Sel.X) and q[Sel.S] == ()


def test_has_path_and_cut():
    p = Sel.parents_of(CHAIN_MASK, 0)
    assert Sel.has_path(p, Sel.X, Sel.Y) and Sel.has_path(p, Sel.Z, Sel.Y) and not Sel.has_path(p, Sel.Y, Sel.X) and not Sel.has_path(p, Sel.W, Sel.Y)
    assert Sel.cut_into_x(p)[Sel.X] == () and Sel.cut_into_x(p)[Sel.Y] == p[Sel.Y]


def test_s_admissibility_of_the_named_cases():
    chain_s = lambda *t: Sel.parents_of(CHAIN_MASK, _s_mask(*t))    # noqa: E731
    assert not Sel.s_admissible(chain_s(Sel.Z)) and Sel.s_admissible(chain_s(Sel.Z), (Sel.Z,)) and Sel.s_admissible(chain_s(Sel.Z), (Sel.Z, Sel.W)) and not Sel.s_admissible(chain_s(Sel.Z), (Sel.W,))
    assert Sel.s_admissible(chain_s(Sel.X)) and Sel.s_admissible(chain_s(Sel.W)) and Sel.s_admissible(chain_s())
    assert not any(Sel.s_admissible(chain_s(Sel.Y), a) for a in Sel.SETS) and not any(Sel.s_admissible(chain_s(Sel.Z, Sel.Y), a) for a in Sel.SETS)


def test_classify_the_named_cases():
    chain_s = lambda *t: Sel.classify(Sel.parents_of(CHAIN_MASK, _s_mask(*t)))    # noqa: E731
    assert chain_s(Sel.Z) == {"kind": "menge", "admissible": ((Sel.Z,), (Sel.Z, Sel.W))} and chain_s(Sel.X)["kind"] == "direkt" and chain_s(Sel.Y) == {"kind": "keine", "admissible": ()}
    assert chain_s()["kind"] == "direkt" and chain_s(Sel.Z, Sel.Y)["kind"] == "keine"
    assert Sel.classify(Sel.parents_of(0b000100, _s_mask(Sel.Y)))["kind"] == "trivial"                   # nur Z → Y: kein Pfeil X → Y


def test_open_paths_explain_a_failure():
    p = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Y))
    paths = Sel.open_paths(p, ())
    assert [Sel.path_text(x) for x in paths] == ["S → Y"]
    q = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z))
    assert {Sel.path_text(x) for x in Sel.open_paths(q, ())} == {"S → Z → Y"}                  # der Pfad über X ist abgeschnitten: Im Diagramm ohne Pfeile in X hat X keine Eltern


def test_open_paths_vanish_with_the_admissible_set():
    q = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z))
    assert Sel.open_paths(q, ()) and Sel.open_paths(q, (Sel.Z,)) == []


def test_binary_hand_example_exactly():
    """Z → Y und X → Y, S → Z; P(Z = 1) = 0,2 (Quelle) und 0,7 (Ziel), X ~ B(0,5); P(Y = 1 | z, x) = (0,3, 0,4, 0,5, 0,8) für (z, x) = (0,0), (0,1), (1,0), (1,1).
    Wirkung je z: 0,1 und 0,3; Quelle 0,8 · 0,1 + 0,2 · 0,3 = 0,14, Ziel 0,3 · 0,1 + 0,7 · 0,3 = 0,24."""
    edge_mask = (1 << Sel.EDGES.index((Sel.Z, Sel.Y))) | (1 << Sel.EDGES.index((Sel.X, Sel.Y)))
    parents = Sel.parents_of(edge_mask, _s_mask(Sel.Z))
    base = {Sel.W: [0.5], Sel.X: [0.5], Sel.Y: [0.3, 0.4, 0.5, 0.8]}
    source, target = {**base, Sel.Z: [0.2]}, {**base, Sel.Z: [0.7]}
    assert Sel.target_effect(parents, source) == pytest.approx(0.14) and Sel.target_effect(parents, target) == pytest.approx(0.24)
    assert Sel.transport_formula(parents, source, target, ()) == pytest.approx(0.14)
    assert Sel.transport_formula(parents, source, target, (Sel.Z,)) == pytest.approx(0.24) and Sel.transport_formula(parents, source, target, (Sel.Z, Sel.W)) == pytest.approx(0.24)
    assert Sel.transport_formula(parents, source, target, (Sel.W,)) == pytest.approx(0.14)


def test_joint_probabilities_sum_to_one():
    parents = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z))
    source, _ = Sel.tables_pair(parents, SplitMix64(1))
    assert sum(Sel._joint(parents, source).values()) == pytest.approx(1.0)
    assert sum(Sel._joint(parents, source, 1).values()) == pytest.approx(1.0) and sum(Sel._joint(parents, source, 0).values()) == pytest.approx(1.0)
    assert all(x == 1 for (_, _, x, _) in Sel._joint(parents, source, 1))


def test_random_tables_have_the_right_sizes_and_range():
    parents = Sel.parents_of(63, _s_mask(Sel.Z, Sel.Y))
    t = Sel.random_tables(parents, SplitMix64(2))
    assert [len(t[v]) for v in Sel.OBSERVED] == [1, 2, 4, 8] and all(0.1 <= q <= 0.9 for v in Sel.OBSERVED for q in t[v])
    assert all(len(Sel.random_tables(parents, SplitMix64(s))[Sel.Y]) == 8 for s in range(3))


def test_tables_pair_differ_exactly_where_s_points():
    parents = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z, Sel.Y))
    source, target = Sel.tables_pair(parents, SplitMix64(3))
    assert (source[Sel.Z] != target[Sel.Z]).all() and (source[Sel.Y] != target[Sel.Y]).all() and (source[Sel.X] == target[Sel.X]).all() and (source[Sel.W] == target[Sel.W]).all()
    assert Sel.tables_pair(parents, SplitMix64(3))[0][Sel.Z] == pytest.approx(source[Sel.Z])


def test_effect_is_zero_without_an_arrow_from_x_to_y():
    parents = Sel.parents_of(0b000100, _s_mask(Sel.Y))
    s, t = Sel.tables_pair(parents, SplitMix64(4))
    assert Sel.target_effect(parents, t) == pytest.approx(0.0, abs=1e-12) and Sel.transport_formula(parents, s, t, ()) == pytest.approx(0.0, abs=1e-12)


def test_direct_transport_of_an_s_arrow_into_x():
    parents = Sel.parents_of(CHAIN_MASK, _s_mask(Sel.X))
    for seed in range(5):
        s, t = Sel.tables_pair(parents, SplitMix64(seed))
        assert Sel.transport_formula(parents, s, t, ()) == pytest.approx(Sel.target_effect(parents, t), abs=1e-12)
        assert Sel.target_effect(parents, s) == pytest.approx(Sel.target_effect(parents, t), abs=1e-12)           # die Wirkung hängt nicht von der Zuteilung ab


def test_classify_text_for_each_kind():
    import utr_visualization as V

    kind, text, paths = V.classify_text(Sel.parents_of(CHAIN_MASK, _s_mask(Sel.X)))
    assert kind == "success" and "direkt übertragbar" in text and paths == []
    kind, text, paths = V.classify_text(Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Z)))
    assert kind == "success" and "nur über eine Menge" in text and "{Z}" in text and paths
    kind, text, paths = V.classify_text(Sel.parents_of(CHAIN_MASK, _s_mask(Sel.Y)))
    assert kind == "error" and "nicht übertragbar" in text and [Sel.path_text(p) for p in paths] == ["S → Y"]
    kind, text, paths = V.classify_text(Sel.parents_of(0b000100, _s_mask(Sel.Y)))
    assert kind == "info" and "keinen Pfeil von X nach Y" in text and paths == []


def test_all_edge_and_s_masks_give_valid_graphs():
    for em, sm in itertools.product(range(64), range(16)):
        p = Sel.parents_of(em, sm)
        assert len(p) == 5 and p[Sel.S] == () and all(a != Sel.S or v in Sel.OBSERVED for v, ps in enumerate(p) for a in ps) and all(a < v for v, ps in enumerate(p) for a in ps)
