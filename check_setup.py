"""
Quick sanity check: run this after installing requirements and before
training/deploying, to confirm data files and (optionally) trained
models are present.

Usage:
    python check_setup.py
"""

import os

from src.config import DATASETS, MODELS_DIR

print("Checking dataset files...")
all_data_ok = True
for name, cfg in DATASETS.items():
    exists = os.path.exists(cfg["csv"])
    status = "OK" if exists else "MISSING"
    if not exists:
        all_data_ok = False
    print(f"  [{status}] {cfg['csv']}")

print("\nChecking trained model files...")
all_models_ok = True
for name in DATASETS:
    path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
    exists = os.path.exists(path)
    status = "OK" if exists else "NOT TRAINED YET"
    if not exists:
        all_models_ok = False
    print(f"  [{status}] {path}")

print()
if not all_data_ok:
    print("-> Some data files are missing. See README.md 'Getting the data' section.")
if not all_models_ok:
    print("-> Some models aren't trained yet. Run: python -m src.train --dataset all")
if all_data_ok and all_models_ok:
    print("-> Everything looks good. You can run: streamlit run app/streamlit_app.py")
