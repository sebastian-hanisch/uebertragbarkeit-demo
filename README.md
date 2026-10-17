# Übertragbarkeit: Gilt die Wirkung auch in einer anderen Standort-Gruppe? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-uebertragbarkeit-demo.streamlit.app/)**

Zwölftes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo): Die Stücke davor betrachten **eine** Gruppe. Hier gibt es eine **Quelle** mit Messungen und eine **Zielgruppe**, in der der Faktor eingeführt werden soll: Wann gilt die Wirkung dort, und wann braucht es neue Daten?

## Kernfrage

Der hohe Faktor wirkt stärker, wo die Nachfrage stärker schwankt. Im Ziel schwankt sie stärker als in der Quelle. Die naive Übertragung der mittleren Wirkung liegt daneben; die Umgewichtung über die Verteilung der Schwankung im Ziel nicht. Unterscheidet sich aber der **Mechanismus**, hilft keine Rechnung. Welches Selektionsdiagramm liegt vor, und was kostet geringe Überlappung?

## Modell

- **Vehikel:** die Welt aus Stück 1 und 11 mit **Quelle und Ziel**. Y = 50 + 2 Z + τ(Z) X + 3 ε, τ(z) = 2 + τ₁ z + τ₂ z²; die Zuteilung X = 1[0,5 (Z − Mitte) + η > 0] ist beobachtet. Quelle Z ~ N(0, 1), Ziel Z ~ N(Δ, 1); im Ziel kann außerdem eine unbeobachtete **Zusatzwirkung γ** und ein **Niveauunterschied** wirken. Die **Wahrheit je Zielgruppe** folgt aus dem Modell: τ₀ + τ₁ Δ + τ₂ (Δ² + 1) + γ.
- **Daten:** 500 Standorte in der Quelle (mit Faktor, Servicegrad und Z), 500 Standorte der Zielgruppe mit **nur Z**; `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).
- **Selektionsdiagramm** (`utr_selection.py`): vier beobachtete Größen Z, W, X, Y und der Selektionsknoten S, der auf die Größen zeigt, in denen sich Quelle und Ziel unterscheiden.

## Methodik

- **Wirkungsfunktion** der Quelle: Regression des Servicegrads auf Faktor, Z und die Wechselwirkung (linear oder quadratisch).
- **Naive Übertragung:** die mittlere Wirkung der Quelle, unverändert. **Transportiert:** die Wirkungsfunktion über die gemessenen Momente von Z im Ziel gemittelt, Standardfehler nach der Delta-Methode (Koeffizienten und Momente). **Gewichtet:** dieselben Momente über die Quellstichprobe mit den Gewichten Dichte im Ziel durch Dichte in der Quelle (aus den bekannten Dichten; in der Praxis geschätzt); **effektive Stichprobe** (Σw)² / Σw².
- **S-Zulässigkeit:** Y ist von S getrennt gegeben X und eine Menge Z im Diagramm ohne Pfeile in X (`dsep_graph.py`, wortgleich aus den Vorgängern). Dann gilt die **Transportformel** P*(y | do(x)) = Σ_z P(y | do(x), z) P*(z); mit der leeren Menge ist die Wirkung direkt übertragbar.
- **Pilotstudie:** eine kleine randomisierte Studie im Ziel als Test der Übertragung.
- **Exakte Gegenprobe:** für alle 1 024 Selektionsdiagramme mit vier binären Größen wird die Transportformel gegen die wahre Zielwirkung gerechnet (Tabellen exakt, kein Zufall in der Auswertung).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wahrheit im Ziel = 2 + τ₁ Δ (τ₂ = 0, γ = 0), 300 Wiederholungen.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (Δ = 1, τ₁ = 1,5, Seed 35) | Wahrheit im Ziel **3,50** (Quelle 2,00). Naive Übertragung **2,27** (Fehler −1,23, verfehlt), transportiert **3,74** (Fehler +0,24, Standardfehler 0,421, deckt), gewichtet 3,68 bei einer effektiven Stichprobe von 42,8 % (exakt 36,8 %). | `test_default_sample` |
| Was kostet die naive Übertragung, und was holt die Umgewichtung? (16 Zellen aus Δ = 0 / 0,5 / 1 / 2 und τ₁ = 0 / 0,75 / 1,5 / 3) | Die naive Übertragung liegt um **−τ₁ Δ** daneben (geschlossene Form): −0,75 (Δ = 0,5, τ₁ = 1,5), **−1,49** (Δ = 1, τ₁ = 1,5), −3,00 (Δ = 1, τ₁ = 3), **−6,00** (Δ = 2, τ₁ = 3); ohne Verschiebung oder ohne Modifikator 0. **Transportiert:** Verzerrung höchstens **0,04**, **Deckung 93 bis 97 %** in allen 16 Zellen; die Streuung wächst von 0,28 (Δ = 0) über 0,43 (Δ = 1, τ₁ = 1,5) auf 0,63 (Δ = 2, τ₁ = 3). | `test_experiment_1_shift_and_modifier` |
| Und die Gewichtung? | Bei Δ = 2, τ₁ = 3 verzerrt (**−0,27**) und streut **1,15** gegen 0,63 beim transportierten Schätzer: Der Preis der Gewichte (die Zielstichprobe misst Z direkt, die Gewichte brauchen die Dichten). | `test_experiment_1_shift_and_modifier` |
| Überträgt sich ein Niveauunterschied? (Ziel um 0 / 5 / 20 höher) | Die übertragene Wirkung bleibt **2,01 / 2,01 / 1,97** (Wahrheit 2,00), der mittlere Servicegrad im Ziel liegt um 0,00 / 5,00 / **19,98** höher: Die Wirkung überträgt sich, das Niveau nicht. | `test_experiment_2_level_and_additional_effect` |
| Und eine unbeobachtete Zusatzwirkung γ im Ziel? (γ = 0 / 0,5 / 1 / 2) | Fehler der Übertragung **−0,02 / −0,49 / −0,98 / −1,98** (= −γ), Deckung **95 / 61 / 9 / 0 %**. | `test_experiment_2_level_and_additional_effect` |
| Merkt eine kleine Zielstudie es? (randomisiert, 60 / 200 / 600 Standorte; Anteil der Läufe, in denen sie der Übertragung widerspricht) | γ = 0: 9 / 4 / 5 %; γ = 0,5: 7 / 10 / 15 %; γ = 1: 16 / 33 / **58 %**; γ = 2: **44** / 86 / **98 %**. Mit 60 Standorten ist sie ein schwacher Test. | `test_experiment_2_level_and_additional_effect` |
| Was kostet geringe Überlappung? (Δ = 0,5 / 1 / 2 / 3) | Die effektive Stichprobe der Gewichte ist **0,78 / 0,37 / 0,018 / 0,0001** von n (exakt exp(−Δ²); aus einer Stichprobe 0,78 / 0,37 / 0,020 / 0,0005). | `test_experiment_3_overlap_and_extrapolation` |
| Und die Extrapolation? (gekrümmter Modifikator τ₂ = 0,5; Wahrheit 3,13 / 4,00 / 6,50 / 10,00) | **Lineares** Wirkungsmodell: Verzerrung −0,09 / −0,29 / **−1,37 / −3,50**, Streuung 0,33 / 0,44 / 0,66 / 0,95. **Quadratisches** (die richtige Form): Verzerrung −0,02 / −0,06 / +0,09 / −0,07, Streuung 0,36 / 0,56 / **1,14 / 2,28**: Verzerrung gegen Streuung. | `test_experiment_3_overlap_and_extrapolation` |
| Wann ist eine Wirkung übertragbar? (alle 1 024 Selektionsdiagramme mit vier binären Größen, je sechs Zufallstabellen) | **512** haben keinen Pfeil X → Y (die Wirkung ist überall null). Von den **512** mit Pfeil: **direkt übertragbar 136 (26,6 %)**, **nur über eine zulässige Menge 120 (23,4 %)**, **nicht übertragbar 256 (50 %)**. Ist die S-Zulässigkeit erfüllt, weicht die Formel in **0 von 4 560** Fällen von der wahren Zielwirkung ab; ist sie nicht erfüllt, trifft sie in **0 von 7 728** Fällen trotzdem: hinreichend und (für diese Klasse, generisch) notwendig. Zeigt S auf Y, ist die Wirkung nie übertragbar. | `test_experiment_4_selection_diagrams` |

Handwerte des Textes (`test_hand_values_of_the_text`): τ(z) = 2 + 1,5 z, Quelle N(0, 1), Ziel N(1, 1): Wahrheit **3,5**, naiv **2,0**; mit τ₂ = 0,5 und τ₁ = 1: **4,0**; ESS-Anteil bei Δ = 1: **0,3679**; binäres Beispiel (P(Z = 1) = 0,2 in der Quelle und 0,7 im Ziel): naive Übertragung 0,14, Transportformel **0,24** gleich der wahren Zielwirkung.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Selektionsdiagramm ist bekannt** | Wo S wirkt, ist eine Annahme über die Gruppen; ein falsches Diagramm gibt ein sicheres, aber falsches Urteil. | Wie falsch darf das Diagramm sein? |
| **Z ist in beiden Gruppen gemessen** | Ohne Z im Ziel bleibt nur die naive Übertragung (Experiment 1). | nicht gebaut |
| **Der Mechanismus ist gleich** | Eine unbeobachtete Zusatzwirkung im Ziel verfälscht die Übertragung um genau ihre Größe; eine kleine Zielstudie bemerkt es selten (Experiment 2). | Sensitivität und Falsifikation |
| **Die Gruppen überlappen** | Bei geringer Überlappung schrumpft die effektive Stichprobe der Gewichte wie exp(−Δ²) (Experiment 3). | nicht gebaut |
| **Die Form der Wirkungsfunktion ist bekannt** | Ein zu einfaches Modell extrapoliert verzerrt, ein flexibles streut (Experiment 3). | Schätzen (Stück 6) |
| **Alle Größen sind beobachtet** | Verborgene Größen und S-Pfeile in verborgene Knoten sind nicht gebaut. | Wann geht es gar nicht? |
| **Eine Quelle, ein Ziel** | Datenfusion mehrerer Studien, Meta-Analyse und Selektion in die Stichprobe sind eigene Themen. | nicht gebaut |

Die Verzerrung der naiven Übertragung −τ₁ Δ ist eine **geschlossene Form** (die mittlere Wirkung ist linear in der Mitte von Z), kein Befund; der Befund ist, dass die Umgewichtung sie vollständig beseitigt, was sie an Streuung kostet und dass sie bei einem anderen Mechanismus nichts ausrichtet. Die **512 Diagramme ohne Pfeil X → Y** sind Setup: Dort ist die Wirkung in jeder Gruppe null und jede Formel trifft; in einer ersten Fassung der Gegenprobe zählte das als „Kriterium falsch, Formel trotzdem richtig“ in der Hälfte der Fälle und verdeckte, dass das Kriterium notwendig ist. Das quadratische Modell **kennt** die Form des Modifikators; in der Praxis weiß man sie nicht. Die Zahlen der Pilotstudie hängen an ihrer Größe, die Richtung (klein heißt kaum Widerspruch) ist die Aussage. Das Kriterium und die Transportformel sind aus dem Gedächtnis geschrieben und durch die exakte Gegenprobe für diese Klasse belegt (vier binäre Größen, alle beobachtet), nicht allgemein; die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft.

## Bezug zu den Nachbarn im Portfolio

- [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo) und [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo): enden mit „Übertragbarkeit“ als „Wer setzt an“ für „Die Wirkung ist für alle gleich“ und „Keine Selektion, kein Transport“; hier die Antwort.
- [d-separation-demo](https://github.com/sebastian-hanisch/d-separation-demo), [backdoor-demo](https://github.com/sebastian-hanisch/backdoor-demo): `dsep_graph.py` und das Muster „Diagramm ohne Pfeile in x“ kommen wortgleich bzw. in derselben Form mit; S ist ein weiterer Knoten im Diagramm.
- [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo): dieselbe Welt mit der Nachfrageschwankung Z; hier ist sie Effektmodifikator und unterscheidet sich zwischen den Gruppen.
- **Abgrenzung zu heterogenen Wirkungen:** Die Wirkungsfunktion τ(z) dient hier nur als Mittel für die Umgewichtung; Schätzer für heterogene Wirkungen sind nicht gebaut.

## Tests

Hand-Instanzen (exakte Daten mit τ(z) = 2 + 1,5 z und gekrümmt, Momente und Standardfehler von Hand, effektive Stichprobe (Σw)²/Σw² = 16/6, das binäre Beispiel mit 0,14 und 0,24), Wahrheit je Gruppe gegen den Mittelwert der Wirkungsfunktion und gegen eine randomisierte Simulation, die S-Zulässigkeit der benannten Fälle, die Transportformel gegen die wahre Zielwirkung über **alle 1 024 Diagramme** (hinreichend und notwendig), Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Modellform, Selektionsdiagramm zum Bauen, Permalink, PDF, der Versuch auf Knopfdruck), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py` | d-Separation (wortgleich aus Stück 2) |
| `utr_world.py` | Vehikel: Quelle und Ziel, Wirkungsfunktion, Wahrheit je Gruppe, Gewichte |
| `utr_estimators.py` | Wirkungsfunktion, naive, transportierte und gewichtete Wirkung, effektive Stichprobe, Pilotstudie |
| `utr_selection.py` | Selektionsdiagramme: S-Zulässigkeit, Urteil, offene Pfade, exakte Transportformel für binäre Größen |
| `utr_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `utr_visualization.py`, `utr_pdf_export.py` | Figuren, Tabellen, PDF |
| `utr_constants.py`, `utr_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Datenfusion mehrerer Quellen, Meta-Analyse, Selektion in die Stichprobe:** eigene Themen; hier eine Quelle und ein Ziel.
- **Verborgene Größen, S-Pfeile in verborgene Knoten:** alle Größen sind beobachtet.
- **Geschätzte Gewichte** (Dichtequotienten aus den Daten, Klassifikatoren): die Gewichte kommen aus den bekannten Dichten.
- **Schätzer für heterogene Wirkungen** (CATE, kausale Wälder): die Wirkungsfunktion ist nur ein Mittel für die Umgewichtung.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
