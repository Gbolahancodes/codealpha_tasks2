"""
Streamlit app: Disease Prediction from Medical Data.

Lets a user pick a disease (Heart Disease, Diabetes, Breast Cancer),
enter patient data through a form, and get a prediction + probability
from the best-performing trained model for that dataset.

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st

# Make sure `src` is importable regardless of where streamlit is launched from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASETS
from src.data_utils import load_dataset, get_feature_columns
from src.predict import predict_single, load_model_bundle


st.set_page_config(
    page_title="Disease Prediction from Medical Data",
    page_icon="",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_bundle(dataset_name: str):
    return load_model_bundle(dataset_name)


def render_input_form(dataset_name: str) -> dict:
    """
    Build the input widgets for a dataset. For 'heart' and 'diabetes' this
    uses the hand-labeled field definitions in config.py. For 'breast_cancer'
    (30 numeric features) it auto-generates number inputs from the data's
    own column statistics, since a hand-written form for 30 fields isn't
    practical to maintain.
    """
    cfg = DATASETS[dataset_name]
    values = {}

    if cfg["features"] == "auto":
        df = load_dataset(dataset_name)
        feature_cols = get_feature_columns(dataset_name, df)
        st.caption(
            "This dataset has 30 numeric measurements from a cell nuclei image. "
            "Defaults below are pre-filled with dataset averages -- adjust any "
            "value based on the patient's actual lab report."
        )
        cols_per_row = 3
        rows = [feature_cols[i:i + cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]
        for row_feats in rows:
            cols = st.columns(len(row_feats))
            for col, feat in zip(cols, row_feats):
                default_val = float(round(df[feat].mean(), 3))
                min_val = float(round(df[feat].min(), 3))
                max_val = float(round(df[feat].max() * 1.2, 3))
                values[feat] = col.number_input(
                    feat.replace("_", " ").title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"{dataset_name}_{feat}",
                )
        return values

    # Hand-defined feature form (heart / diabetes)
    feature_defs = cfg["features"]
    feat_items = list(feature_defs.items())
    cols_per_row = 2
    rows = [feat_items[i:i + cols_per_row] for i in range(0, len(feat_items), cols_per_row)]

    for row_feats in rows:
        cols = st.columns(len(row_feats))
        for col, (feat_key, meta) in zip(cols, row_feats):
            if meta["type"] == "select":
                label_choice = col.selectbox(
                    meta["label"], list(meta["options"].keys()), key=f"{dataset_name}_{feat_key}"
                )
                values[feat_key] = meta["options"][label_choice]
            else:
                step = meta.get("step", 1)
                values[feat_key] = col.number_input(
                    meta["label"],
                    min_value=meta["min"],
                    max_value=meta["max"],
                    value=meta["default"],
                    step=step,
                    key=f"{dataset_name}_{feat_key}",
                )
    return values


def render_result(dataset_name: str, result: dict):
    cfg = DATASETS[dataset_name]
    pred = result["prediction"]
    proba = result["probability"]

    label = cfg["positive_label"] if pred == 1 else cfg["negative_label"]

    if pred == 1:
        st.error(f"**Result: {label}**")
    else:
        st.success(f"**Result: {label}**")

    if proba is not None:
        # proba is P(class=1); show confidence in the predicted class
        confidence = proba if pred == 1 else (1 - proba)
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

    st.caption(
        "This is a machine learning estimate based on patterns in historical "
        "data, not a medical diagnosis. Always consult a qualified healthcare "
        "professional for an actual diagnosis and treatment decisions."
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("Disease Prediction from Medical Data")
st.write(
    "Predict the likelihood of a disease from structured patient data using "
    "classification models trained on public medical datasets."
)

dataset_labels = {k: v["display_name"] for k, v in DATASETS.items()}
dataset_name = st.selectbox(
    "Select a disease to screen for:",
    options=list(dataset_labels.keys()),
    format_func=lambda k: dataset_labels[k],
)

try:
    bundle = get_bundle(dataset_name)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.expander("About this model"):
    st.write(f"**Best model:** {bundle['model_name']}")
    st.write("**Test-set performance:**")
    st.json(bundle["metrics"])

st.subheader("Patient Data")
with st.form(key=f"form_{dataset_name}"):
    input_values = render_input_form(dataset_name)
    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_single(dataset_name, input_values)
    st.subheader("Prediction")
    render_result(dataset_name, result)

st.divider()
st.caption(
    "Datasets: UCI Heart Disease (Cleveland), Pima Indians Diabetes, "
    "Wisconsin Breast Cancer (UCI ML Repository via scikit-learn)."
)
