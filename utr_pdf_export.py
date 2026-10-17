"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from utr_evaluation import ROW_NAMES, Analysis, de, pct, signed, verdict_instance

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "₁": "1", "₂": "2", "²": "2", "Δ": "Delta", "τ": "tau", "γ": "gamma", "≤": "<=", "≥": ">="}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Übertragbarkeit: Gilt die Wirkung in einer anderen Gruppe?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Verschiebung von Z im Ziel Δ = {de(s.delta, 2)}; Wirkung τ(z) = 2 + {de(s.tau1, 2)} z + {de(s.tau2, 2)} z²; Zusatzwirkung im Ziel γ = {de(s.gamma, 2)}; Niveauunterschied {de(s.shift, 0)}; "
                            f"Quelle n = {s.ns}, Ziel n = {s.nt} (nur Z), Seed {s.seed}; Wirkungsmodell {s.model}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.6, _s(f"Wahre mittlere Wirkung: Quelle {de(a.source_truth, 2)}, Ziel {de(a.truth, 2)}."), new_x="LMARGIN", new_y="NEXT")
    for r in a.rows:
        pdf.multi_cell(0, 4.6, _s(f"{r.name}: {signed(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}], Fehler {signed(r.error, 3)}, {'deckt' if r.covers else 'deckt nicht'}."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 4.6, _s(f"Effektive Stichprobe der Gewichte: {pct(a.ess_ratio, 1)} (exakt {pct(a.ess_exact, 1)})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: uebertragbarkeit-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
