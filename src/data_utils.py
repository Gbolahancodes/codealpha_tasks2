"""
Shared data loading and preprocessing helpers used by both training
and inference (the Streamlit app), so the two never drift apart.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATASETS, RANDOM_STATE, TEST_SIZE


def load_dataset(name: str) -> pd.DataFrame:
    """Load a registered dataset by key ('heart', 'diabetes', 'breast_cancer')."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}")
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["csv"])
    return df


def get_feature_columns(name: str, df: pd.DataFrame = None) -> list:
    """
    Return the ordered list of feature column names for a dataset.
    For breast_cancer, features are 'auto' -> derived from the CSV
    (every column except the target).
    """
    cfg = DATASETS[name]
    if cfg["features"] == "auto":
        if df is None:
            df = load_dataset(name)
        return [c for c in df.columns if c != cfg["target_col"]]
    return list(cfg["features"].keys())


def split_features_target(name: str, df: pd.DataFrame):
    cfg = DATASETS[name]
    feature_cols = get_feature_columns(name, df)
    X = df[feature_cols].copy()
    y = df[cfg["target_col"]].copy()
    return X, y, feature_cols


def train_test_split_scaled(X: pd.DataFrame, y: pd.Series):
    """
    Split then scale. Scaler is fit ONLY on the training set to avoid
    data leakage, and reused to transform the test set + future inputs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
