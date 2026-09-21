"""
Train and evaluate SVM, Logistic Regression, Random Forest, and XGBoost
on a chosen dataset, then save the best-performing model + its scaler
to /models so the Streamlit app can load them at inference time.

Usage:
    python -m src.train --dataset heart
    python -m src.train --dataset diabetes
    python -m src.train --dataset breast_cancer
    python -m src.train --dataset all          # trains all three
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import DATASETS, MODELS_DIR, RANDOM_STATE
from src.data_utils import load_dataset, split_features_target, train_test_split_scaled


def build_models():
    """
    Fresh, unfitted model instances. Kept in a function (not module-level)
    so every call to train_one_dataset starts from a clean model.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    return metrics


def train_one_dataset(name: str, verbose: bool = True) -> dict:
    cfg = DATASETS[name]
    df = load_dataset(name)
    X, y, feature_cols = split_features_target(name, df)
    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y)

    results = {}
    fitted_models = {}

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[model_name] = metrics
        fitted_models[model_name] = model
        if verbose:
            print(f"  [{name}] {model_name:<20} -> {metrics}")

    # Pick the best model by F1 score (balances precision/recall --
    # more meaningful than raw accuracy for medical data, since classes
    # are often imbalanced and false negatives matter).
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_model = fitted_models[best_model_name]

    if verbose:
        print(f"  [{name}] BEST MODEL -> {best_model_name} ({results[best_model_name]})")

    # Persist everything the app needs for inference
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_name": best_model_name,
        "metrics": results[best_model_name],
        "dataset": name,
    }
    out_path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
    joblib.dump(bundle, out_path)

    # Also save a human-readable metrics report for the README / app
    report_path = os.path.join(MODELS_DIR, f"{name}_metrics.json")
    with open(report_path, "w") as f:
        json.dump(
            {"best_model": best_model_name, "all_results": results},
            f,
            indent=2,
        )

    if verbose:
        print(f"  [{name}] saved -> {out_path}")
        print(f"  [{name}] saved -> {report_path}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train disease prediction models.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to train on (default: all)",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"Training on: {targets}\n")
    for name in targets:
        print(f"=== {DATASETS[name]['display_name']} ===")
        train_one_dataset(name)

    print("Done. Trained models are saved in the models/ directory.")


if __name__ == "__main__":
    main()
