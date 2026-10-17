"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen. Die Modellform ist eine Auswahl (Adresszeile: Index 0 oder 1)."""

import random
from dataclasses import dataclass

import streamlit as st

import utr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


SETTING_SPECS = {
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX, C.DELTA_STEP),
    "tau1_slider": SettingSpec("t1", float, C.DEFAULT_TAU1, C.TAU1_MIN, C.TAU1_MAX, C.TAU1_STEP),
    "tau2_slider": SettingSpec("t2", float, C.DEFAULT_TAU2, C.TAU2_MIN, C.TAU2_MAX, C.TAU2_STEP),
    "gamma_slider": SettingSpec("g", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX, C.GAMMA_STEP),
    "shift_slider": SettingSpec("lv", float, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX, C.SHIFT_STEP),
    "ns_slider": SettingSpec("ns", int, C.DEFAULT_NS, C.NS_MIN, C.NS_MAX, C.NS_STEP),
    "nt_slider": SettingSpec("nt", int, C.DEFAULT_NT, C.NT_MIN, C.NT_MAX, C.NT_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
    "model_radio": SettingSpec("m", int, C.DEFAULT_MODEL, 0, len(C.MODELS) - 1, 1, C.MODELS),
}
PRESET_KEYS = {"delta": "delta_slider", "tau1": "tau1_slider", "tau2": "tau2_slider", "gamma": "gamma_slider", "shift": "shift_slider", "ns": "ns_slider", "nt": "nt_slider", "seed": "seed_input",
               "model": "model_radio"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
