"""Gemeinsame Fixtures: die Hand-Instanzen (DEMO-PLAYBOOK §1b) und die einmal berechneten Experimente."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def exact_fit():
    """Mini-Instanz 1: exakte Daten y = 3 + z + (2 + 1,5 z) x (kein Rest). Die Regression y ~ x + z + x z trifft die Koeffizienten (x: 2, x z: 1,5) genau.
    z = (−1, 0, 1, −1, 0, 1), x = (0, 0, 0, 1, 1, 1): τ(−1) = 0,5, τ(0) = 2, τ(1) = 3,5; mittlere Wirkung in der Quelle (z-Mittel 0) 2."""
    z = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return z, x, 3.0 + z + (2.0 + 1.5 * z) * x


@pytest.fixture
def quadratic_fit():
    """Mini-Instanz 2: exakte Daten mit gekrümmter Wirkung τ(z) = 2 + 1,5 z + 0,5 z², z = (−2, −1, 0, 1, 2) für beide Gruppen, y = 1 + z + τ(z) x."""
    z = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 2)
    x = np.repeat([0.0, 1.0], 5)
    return z, x, 1.0 + z + (2.0 + 1.5 * z + 0.5 * z ** 2) * x


@pytest.fixture(scope="session")
def shift_res():
    import utr_evaluation as ev

    return ev.shift_experiment()


@pytest.fixture(scope="session")
def mechanism_res():
    import utr_evaluation as ev

    return ev.mechanism_experiment()


@pytest.fixture(scope="session")
def overlap_res():
    import utr_evaluation as ev

    return ev.overlap_experiment()


@pytest.fixture(scope="session")
def selection_res():
    import utr_evaluation as ev

    return ev.selection_experiment()
