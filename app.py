"""Streamlit app: credit risk assessment demo (educational).

Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import (
    METRICS_PATH,
    MODEL_PATH,
    OPTIONS_PATH,
    load_json,
    load_model,
    risk_band,
    score_applicant,
)
from ui.components import (
    disclaimer_html,
    empty_state_html,
    factor_bars_html,
    footer_html,
    gauge_html,
    hero_html,
)
from ui.labels import FIELD_GROUPS, field_label

st.set_page_config(page_title="Credit Risk Assessment", page_icon="📊", layout="wide")
style = (Path(__file__).parent / "ui" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load the model once and reuse it between interactions."""
    return load_model()


def render_field(name, spec):
    """Draw the right input for one column and return its value."""
    label = field_label(name)
    if spec["type"] == "categorical":
        return st.selectbox(label, spec["choices"], key=name)

    low, high, default = spec["min"], spec["max"], spec["median"]
    if spec["integer"]:
        low, high, default = int(low), int(high), int(round(default))
        if high - low <= 100:
            return st.slider(label, low, high, default, key=name)
        step = 50 if high - low > 1000 else 1
        return st.number_input(label, low, high, default, step=step, key=name)
    return st.number_input(label, float(low), float(high), float(default), key=name)


def build_form(columns):
    """Lay the fields out in sections and return {column: value}."""
    values = {}
    sections = list(FIELD_GROUPS)
    listed = {name for _, names in FIELD_GROUPS for name in names}
    leftover = [name for name in columns if name not in listed]
    if leftover:
        sections.append(("Other details", leftover))

    for title, names in sections:
        names = [name for name in names if name in columns]
        if not names:
            continue
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        for index, name in enumerate(names):
            with (left if index % 2 == 0 else right):
                values[name] = render_field(name, columns[name])
    return values


def show_model_details(metrics):
    """Plain-language facts about how the model was built and tested."""
    data, test = metrics["dataset"], metrics["test"]
    st.write(
        f"Dataset: {data['name']}. {data['rows']} loans, "
        f"{data['bad_risk_share']:.0%} of them bad risks. "
        f"{data['train_rows']} used for training and model choice, "
        f"{data['test_rows']} kept aside for one final test."
    )
    st.write("Cross-validation on the training data (mean ± spread, 'bad risk' as the positive class):")
    rows = {
        model: {
            "ROC-AUC": f"{s['roc_auc']['mean']:.3f} ± {s['roc_auc']['std']:.3f}",
            "Precision": f"{s['precision']['mean']:.3f}",
            "Recall": f"{s['recall']['mean']:.3f}",
            "F1": f"{s['f1']['mean']:.3f}",
        }
        for model, s in metrics["cv_comparison"].items()
    }
    st.table(pd.DataFrame(rows).T)
    st.write(
        f"Selected: **{metrics['selected_model']}**. "
        f"Final test: ROC-AUC {test['roc_auc']:.3f}, precision {test['precision']:.3f}, "
        f"recall {test['recall']:.3f}, F1 {test['f1']:.3f}."
    )
    st.caption(
        "Sex, marital status and nationality were removed on purpose. "
        "The score is not a calibrated probability of default."
    )


if not (MODEL_PATH.exists() and METRICS_PATH.exists() and OPTIONS_PATH.exists()):
    st.error("The trained model is missing. Run `python train_model.py` first, then restart the app.")
    st.stop()

metrics = load_json(METRICS_PATH)
options = load_json(OPTIONS_PATH)

st.markdown(hero_html(metrics), unsafe_allow_html=True)

form_column, result_column = st.columns([3, 2], gap="large")

with form_column:
    with st.form("applicant_form"):
        applicant = build_form(options["columns"])
        submitted = st.form_submit_button("Assess risk", type="primary")

with result_column:
    if not submitted:
        st.markdown(empty_state_html(), unsafe_allow_html=True)
    else:
        try:
            with st.spinner("Scoring..."):
                score = score_applicant(get_model(), applicant)
        except ValueError as error:
            st.error(str(error))
        else:
            st.markdown(gauge_html(score, risk_band(score)), unsafe_allow_html=True)
            st.markdown(factor_bars_html(metrics.get("top_features", [])), unsafe_allow_html=True)

with st.expander("How the model was built and tested"):
    show_model_details(metrics)

st.markdown(disclaimer_html(), unsafe_allow_html=True)
st.markdown(footer_html(), unsafe_allow_html=True)
