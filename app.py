"""Übertragbarkeit: Gilt die Wirkung auch in einer anderen Standort-Gruppe? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes Stück der Linie Kausale Diagramme und Adjustierung. Eine Wirkung aus einer Quellgruppe gilt in einer Zielgruppe, wenn sich beide nur in dem unterscheiden, was man messen kann:
Umgewichtung über die Verteilung des Effektmodifikators (Transportformel), Selektionsdiagramme (S-Zulässigkeit) und die Grenzen: anderer Mechanismus, geringe Überlappung, Extrapolation.
Kern: utr_estimators.py, utr_selection.py, utr_world.py; Diagramme: dsep_graph.py (wortgleich aus den Vorgängern).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import utr_constants as C
import utr_selection as Sel
from utr_evaluation import (
    Settings,
    analyse,
    de,
    mechanism_experiment,
    overlap_experiment,
    pct,
    selection_experiment,
    shift_experiment,
    signed,
    verdict_instance,
    verdict_mechanism,
    verdict_overlap,
    verdict_selection,
    verdict_shift,
)
from utr_pdf_export import build_pdf
from utr_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from utr_visualization import (
    build_effect_chart,
    build_estimates_chart,
    build_estimates_table,
    build_gamma_table,
    build_level_table,
    build_mechanism_chart,
    build_overlap_chart,
    build_overlap_table,
    build_selection_chart,
    build_selection_diagram,
    build_selection_table,
    build_shift_chart,
    build_shift_table,
    build_summary_table,
    classify_text,
    style_estimates_table,
)

st.set_page_config(page_title="Übertragbarkeit – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"
EDGE_KEYS = {e: f"sel_edge_{Sel.NAMES[e[0]]}{Sel.NAMES[e[1]]}" for e in Sel.EDGES}
S_KEYS = {v: f"sel_s_{Sel.NAMES[v]}" for v in Sel.OBSERVED}
SELECTION_EXAMPLES = {
    "S → Z: andere Verteilung von Z": dict(edges=[(Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)], s=[Sel.Z]),
    "S → X: andere Zuteilung": dict(edges=[(Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)], s=[Sel.X]),
    "S → Y: anderer Mechanismus": dict(edges=[(Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)], s=[Sel.Y]),
    "S → Z und S → Y": dict(edges=[(Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)], s=[Sel.Z, Sel.Y]),
    "S → W: W ohne Weg zu Y": dict(edges=[(Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y), (Sel.Z, Sel.W)], s=[Sel.W]),
}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _shift():
    return shift_experiment()


@st.cache_data(show_spinner=SPINNER)
def _mechanism():
    return mechanism_experiment()


@st.cache_data(show_spinner=False)
def _overlap():
    return overlap_experiment()


@st.cache_data(show_spinner=SPINNER)
def _selection():
    return selection_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def set_selection_example(name):
    ex = SELECTION_EXAMPLES[name]
    for e, key in EDGE_KEYS.items():
        st.session_state[key] = e in ex["edges"]
    for v, key in S_KEYS.items():
        st.session_state[key] = v in ex["s"]


st.title("🌍 Übertragbarkeit")
st.markdown(
    """
**Gilt die Wirkung auch woanders?** Dies ist das zwölfte Stück der Linie **Kausale Diagramme und Adjustierung**. Die Stücke davor haben **eine** Gruppe von Standorten betrachtet; hier gibt es eine **Quelle** mit Messungen
und eine **Zielgruppe**, in der der Faktor eingeführt werden soll. Eine Wirkung gilt dort, wenn sich beide Gruppen nur in dem unterscheiden, was man messen kann: Dann genügt es, die Wirkung der Quelle über die Verteilung
im Ziel neu zu mitteln. Unterscheidet sich der **Mechanismus** selbst, hilft keine Rechnung. Ein **Selektionsdiagramm** sagt, welcher Fall vorliegt.

Weiter unten: das **Selektionsdiagramm zum Bauen**, vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Die Wirkung hängt von Z ab.** Der hohe Faktor wirkt stärker, wo die Nachfrage stärker schwankt (Steigung τ₁). Die **mittlere Wirkung einer Gruppe** hängt deshalb davon ab, wie Z dort verteilt ist.
2. **Naive Übertragung:** die mittlere Wirkung der Quelle, unverändert für das Ziel übernommen. Sie liegt um **τ₁ mal Verschiebung** daneben.
3. **Transportiert:** die Wirkungsfunktion der Quelle, über die gemessene Verteilung von Z im Ziel gemittelt. Dazu braucht es Z in **beiden** Gruppen.
4. **Gewichtet:** dasselbe über die Quellstichprobe mit Gewichten (Dichte im Ziel durch Dichte in der Quelle). Die **effektive Stichprobe** zeigt, was geringe Überlappung kostet.
5. **Zwei Dinge lassen sich nicht übertragen:** ein **Niveauunterschied** (die Wirkung bleibt) ist harmlos, eine **unbeobachtete Zusatzwirkung** im Ziel verfälscht die Übertragung um genau ihre Größe.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Verschiebung und Modifikator**")
    delta = st.slider("Verschiebung Δ der Nachfrageschwankung im Ziel", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider", help="Mittel von Z im Ziel; in der Quelle 0 (beide mit Streuung 1).")
    tau1 = st.slider("Stärke des Modifikators τ₁", *bounds("tau1_slider"), step=C.TAU1_STEP, key="tau1_slider", help="Wie stark die Wirkung des Faktors mit Z steigt: τ(z) = 2 + τ₁ z + τ₂ z².")
    tau2 = st.slider("Krümmung des Modifikators τ₂", *bounds("tau2_slider"), step=C.TAU2_STEP, key="tau2_slider", help="0: linear. Größer: gekrümmt; das lineare Wirkungsmodell extrapoliert dann falsch.")
    st.markdown("**Was sich nicht übertragen lässt**")
    gamma = st.slider("Zusatzwirkung γ im Ziel (unbeobachtet)", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider", help="S wirkt auf y: Im Ziel wirkt der Faktor um γ stärker, aus Gründen, die nicht gemessen sind.")
    shift = st.slider("Niveauunterschied im Ziel", *bounds("shift_slider"), step=C.SHIFT_STEP, key="shift_slider", help="Der mittlere Servicegrad im Ziel liegt um so viel höher; die Wirkung des Faktors bleibt gleich.")
    st.markdown("**Daten und Modell**")
    ns = st.slider("Standorte in der Quelle", *bounds("ns_slider"), step=C.NS_STEP, key="ns_slider", help="Mit Faktor, Servicegrad und Z gemessen.")
    nt = st.slider("Standorte der Zielgruppe (nur Z)", *bounds("nt_slider"), step=C.NT_STEP, key="nt_slider", help="Im Ziel ist nur Z gemessen, nicht der Faktor und nicht der Servicegrad.")
    model = st.radio("Wirkungsmodell der Quelle", C.MODELS, key="model_radio", help="Linear: τ(z) = b₀ + b₁ z. Quadratisch: mit z². Die richtige Form kennt man nicht.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"delta_slider": round(float(delta), 2), "tau1_slider": round(float(tau1), 2), "tau2_slider": round(float(tau2), 2), "gamma_slider": round(float(gamma), 2),
                   "shift_slider": round(float(shift), 1), "ns_slider": int(ns), "nt_slider": int(nt), "seed_input": int(seed), "model_radio": model})

settings = Settings(round(float(delta), 2), round(float(tau1), 2), round(float(tau2), 2), round(float(gamma), 2), round(float(shift), 1), int(ns), int(nt), int(seed), model)
a = _analysis(settings)
by = {r.name: r for r in a.rows}

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Wirkung im Ziel", de(a.truth, 2), delta=f"in der Quelle {de(a.source_truth, 2)}", delta_color="off", help="Mittlere Wirkung des Faktors in der Zielgruppe, aus dem Modell.")
for col, name, help_text in ((c2, "Naive Übertragung", "Die mittlere Wirkung der Quelle, unverändert für das Ziel übernommen."),
                             (c3, "Transportiert (Z im Ziel gemessen)", "Die Wirkungsfunktion der Quelle, über die gemessene Verteilung von Z im Ziel gemittelt (Delta-Intervall)."),
                             (c4, "Gewichtet (bekannte Dichten)", "Die Wirkungsfunktion über die Quellstichprobe mit den Gewichten Dichte im Ziel durch Dichte in der Quelle.")):
    r = by[name]
    col.metric(name, signed(r.estimate, 2), delta=f"Fehler {signed(r.error, 2)}, Intervall {de(r.lo, 2)} bis {de(r.hi, 2)}", delta_color="off", help=help_text)

left, right = st.columns([1.15, 1])
with left:
    st.plotly_chart(build_effect_chart(a), width="stretch", key="effect_chart")
    st.caption("Die Wirkung des Faktors in Abhängigkeit von Z: wahr in der Quelle (gestrichelt), wahr im Ziel (orange, mit der Zusatzwirkung) und geschätzt aus der Quelle (grün). Flächen: die Verteilungen von Z.")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt mit Intervall: grün, wenn es die Wahrheit im Ziel deckt, rot, wenn nicht. Orange senkrecht: Wahrheit im Ziel; grau gepunktet: Wahrheit in der Quelle.")
verdict(verdict_instance(a))
st.dataframe(style_estimates_table(build_estimates_table(a)), hide_index=True, width="stretch")
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="uebertragbarkeit.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Selektionsdiagramm zum Bauen ----------------------------------------------------------------------------------------------------------

st.subheader("🧩 Selektionsdiagramm zum Bauen")
st.caption("Vier beobachtete Größen: Z und W (Kovariaten), X (Faktor) und Y (Servicegrad). Der Knoten **S** (Raute) zeigt auf die Größen, in denen sich Quelle und Ziel unterscheiden. Die Wirkung ist direkt übertragbar, wenn Y gegeben X "
           "von S getrennt ist (im Diagramm ohne Pfeile in X); sonst vielleicht über eine Menge von Kovariaten, die man im Ziel messen kann.")
for k, key in EDGE_KEYS.items():
    st.session_state.setdefault(key, k in ((Sel.Z, Sel.X), (Sel.Z, Sel.Y), (Sel.X, Sel.Y)))
for v, key in S_KEYS.items():
    st.session_state.setdefault(key, v == Sel.Z)
ex_cols = st.columns(len(SELECTION_EXAMPLES))
for col, name in zip(ex_cols, SELECTION_EXAMPLES):
    with col:
        st.button(name, width="stretch", on_click=set_selection_example, args=(name,), key=f"sel_example_{name}")
edge_cols = st.columns(len(Sel.EDGES))
for col, e in zip(edge_cols, Sel.EDGES):
    with col:
        st.checkbox(f"{Sel.NAMES[e[0]]} → {Sel.NAMES[e[1]]}", key=EDGE_KEYS[e])
s_cols = st.columns(len(Sel.OBSERVED))
for col, v in zip(s_cols, Sel.OBSERVED):
    with col:
        st.checkbox(f"S → {Sel.NAMES[v]}", key=S_KEYS[v])
edge_mask = sum(1 << i for i, e in enumerate(Sel.EDGES) if st.session_state[EDGE_KEYS[e]])
s_mask = sum(1 << i for i, v in enumerate(Sel.OBSERVED) if st.session_state[S_KEYS[v]])
parents = Sel.parents_of(edge_mask, s_mask)
kind, text, paths = classify_text(parents)
cl = Sel.classify(parents)
sel_left, sel_right = st.columns([1, 1])
with sel_left:
    chosen = min((m for m in cl["admissible"] if m), key=len) if cl["kind"] == "menge" else ()
    st.plotly_chart(build_selection_diagram(parents, chosen), width="stretch", key="selection_chart")
with sel_right:
    getattr(st, kind)(text)
    if paths:
        st.markdown("**Offene Pfade zwischen S und Y** (gegeben X und die genannte Menge):")
        for p in paths[:6]:
            st.markdown(f"- {Sel.path_text(p)}")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Verschiebung und Modifikator")
res1 = _shift()
st.caption(f"{C.EXP_REPS} feste Wiederholungen je Zelle, {C.EXP_N} Standorte in der Quelle, {C.EXP_NT} Standorte der Zielgruppe (nur Z). Die Wahrheit im Ziel ist 2 + τ₁ Δ. Die naive Übertragung ist zugleich der Fall, in dem Z im Ziel nicht gemessen ist.")
st.plotly_chart(build_shift_chart(res1), width="stretch", key="shift_chart")
verdict(verdict_shift(res1))
st.dataframe(build_shift_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Was sich nicht übertragen lässt")
res2 = _mechanism()
st.caption(f"{res2['reps']} feste Wiederholungen. Links ein Niveauunterschied im Ziel (die Wirkung bleibt), rechts eine unbeobachtete Zusatzwirkung γ (Fehler −γ); dazu der Anteil der Läufe, in denen eine kleine "
           "randomisierte Studie im Ziel der Übertragung widerspricht.")
st.dataframe(build_level_table(res2), hide_index=True, width="stretch")
st.plotly_chart(build_mechanism_chart(res2), width="stretch", key="mechanism_chart")
verdict(verdict_mechanism(res2))
st.dataframe(build_gamma_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Überlappung und Extrapolation")
res3 = _overlap()
st.caption("Die Zielgruppe liegt um Δ Standardabweichungen neben der Quelle; der Modifikator ist gekrümmt (τ₂ = 0,5). Die effektive Stichprobe der Gewichte fällt wie exp(−Δ²); das lineare Wirkungsmodell extrapoliert verzerrt, "
           "das quadratische (die richtige Form, die man in der Praxis nicht kennt) streut stärker.")
st.plotly_chart(build_overlap_chart(res3), width="stretch", key="overlap_chart")
verdict(verdict_overlap(res3))
st.dataframe(build_overlap_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Alle Selektionsdiagramme (exakt)")
st.caption("Alle 2⁶ Kantenmengen × 2⁴ S-Pfeile für vier binäre Größen (1 024 Diagramme), je sechs Zufallstabellen für Quelle und Ziel. Die Hälfte hat keinen Pfeil von X nach Y (die Wirkung ist überall null); "
           "bei den übrigen wird die Transportformel gegen die wahre Zielwirkung geprüft.")
if st.button("▶ Experiment 4 rechnen (etwa 5 Sekunden)", key="selection_run"):
    st.session_state["selection_ready"] = True
if st.session_state.get("selection_ready"):
    res4 = _selection()
    st.plotly_chart(build_selection_chart(res4), width="stretch", key="selection_summary_chart")
    verdict(verdict_selection(res4))
    st.dataframe(build_selection_table(res4), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Selektionsdiagramm ist bekannt** | Wo S wirkt, ist eine Annahme über die Gruppen; ein falsches Diagramm gibt ein sicheres, aber falsches Urteil. | Wie falsch darf das Diagramm sein? |
| **Z ist in beiden Gruppen gemessen** | Ohne Z im Ziel bleibt nur die naive Übertragung (Experiment 1). | nicht gebaut |
| **Der Mechanismus ist gleich** | Eine unbeobachtete Zusatzwirkung im Ziel verfälscht die Übertragung um genau ihre Größe; eine kleine Zielstudie bemerkt es selten (Experiment 2). | Sensitivität und Falsifikation |
| **Die Gruppen überlappen** | Bei geringer Überlappung schrumpft die effektive Stichprobe der Gewichte wie exp(−Δ²) (Experiment 3). | nicht gebaut |
| **Die Form der Wirkungsfunktion ist bekannt** | Ein zu einfaches Modell extrapoliert verzerrt, ein flexibles streut (Experiment 3). | Schätzen (Stück 6) |
| **Alle Größen sind beobachtet** | Verborgene Größen und S-Pfeile in verborgene Knoten sind nicht gebaut. | Wann geht es gar nicht? |
| **Eine Quelle, ein Ziel** | Datenfusion mehrerer Studien, Meta-Analyse und Selektion in die Stichprobe sind eigene Themen. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wirkungsfunktion.** $\tau(z) = E[Y(1) - Y(0) \mid Z = z]$; hier $\tau(z) = \tau_0 + \tau_1 z + \tau_2 z^2$ (im Ziel $+\gamma$). Mittlere Wirkung einer Gruppe: $E[\tau(Z)]$, in der Quelle mit $Z \sim N(0,1)$ gleich
$\tau_0 + \tau_2$, im Ziel mit $Z \sim N(\Delta,1)$ gleich $\tau_0 + \tau_1 \Delta + \tau_2(\Delta^2 + 1) + \gamma$. Die naive Übertragung liegt um $\tau_1 \Delta + \tau_2 \Delta^2$ daneben.

**Transportformel.** Ist $Z$ in beiden Gruppen gemessen und $S \perp Y \mid X, Z$ im Diagramm ohne Pfeile in $X$ (S-zulässig): $P^*(y \mid \mathrm{do}(x)) = \sum_z P(y \mid \mathrm{do}(x), z)\,P^*(z)$.
Ohne $Z$ ($S \perp Y \mid X$): direkt, $P^*(y \mid \mathrm{do}(x)) = P(y \mid \mathrm{do}(x))$.

**Schätzer.** Regression $Y \sim X + Z + XZ\,(+ Z^2 + XZ^2)$; $\hat\tau(z) = b_0 + b_1 z\,(+ b_2 z^2)$. Transportiert: $\hat b_0 + \hat b_1 \bar z^* (+ \hat b_2 \overline{z^{*2}})$ mit den Momenten der Zielstichprobe, Standardfehler
nach der Delta-Methode aus der Kovarianz der Koeffizienten und der der Momente. Gewichtet: dieselben Momente über die Quellstichprobe mit $w = p^*(z)/p(z) = e^{\Delta z - \Delta^2/2}$.

**Effektive Stichprobe.** $(\sum w)^2/\sum w^2$; für die Gauß-Verschiebung $n\,e^{-\Delta^2}$.

**Literatur.** J. Pearl, E. Bareinboim, *Transportability of causal and statistical relations*, AAAI 2011; dieselben, *External validity: from do-calculus to transportability across populations*, Statistical Science 29, 2014;
E. Bareinboim, J. Pearl, *Causal inference and the data-fusion problem*, PNAS 113, 2016 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
