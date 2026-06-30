"""Page 5: Explainability — feature importance, SHAP, and summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from _shared import FIGURES_DIR, REPORTS_DIR

st.title("Explainability")

EXPL_DIR = FIGURES_DIR / "explainability"

st.markdown(
    "What drives the `will_pit_next_3_laps` predictions. "
    "Run `python src/models/explain_model.py --year 2025` to (re)generate these artifacts."
)

figures = [
    ("Global feature importance", EXPL_DIR / "global_feature_importance.png"),
    ("Permutation importance", EXPL_DIR / "permutation_importance.png"),
    ("SHAP summary", EXPL_DIR / "shap_summary.png"),
]

shown = False
for title, path in figures:
    if path.is_file():
        st.subheader(title)
        st.image(str(path), use_container_width=True)
        shown = True

if not shown:
    st.info("No explainability figures found yet.")

st.subheader("Explainability summary")
report = REPORTS_DIR / "model_explainability_summary.md"
if report.is_file():
    st.markdown(report.read_text())
else:
    st.info("Explainability summary report not found.")
