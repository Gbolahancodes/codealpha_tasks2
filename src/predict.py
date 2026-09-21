"""
Load a trained model bundle and run predictions on new patient input.
Used by the Streamlit app (app/streamlit_app.py).
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR


def load_model_bundle(dataset_name: str) -> dict:
    """Load the {model, scaler, feature_cols, ...} bundle for a dataset."""
    path = os.path.join(MODELS_DIR, f"{dataset_name}_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found for '{dataset_name}'. "
            f"Run `python -m src.train --dataset {dataset_name}` first."
        )
    return joblib.load(path)


def predict_single(dataset_name: str, input_values: dict) -> dict:
    """
    input_values: dict mapping feature_col -> raw value, in any order.
    Returns: {"prediction": 0/1, "probability": float, "label": str}
    """
    bundle = load_model_bundle(dataset_name)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # Enforce correct column order -- this MUST match training order
    row = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)
    row_scaled = scaler.transform(row)

    pred = int(model.predict(row_scaled)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row_scaled)[0][1])

    return {"prediction": pred, "probability": proba}
