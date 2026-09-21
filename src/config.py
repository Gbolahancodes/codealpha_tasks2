"""
Central configuration: paths, dataset definitions, and feature metadata.
Editing this file is the only thing you need to do to point the project
at different data or add a new disease dataset later.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# target_col   : name of the label column in the CSV
# positive_label: human meaning of target == 1
# feature_info : dict used both for training (order of columns) and for
#                building the Streamlit input form (label, type, range)
# ---------------------------------------------------------------------------

DATASETS = {
    "heart": {
        "csv": os.path.join(DATA_DIR, "heart.csv"),
        "target_col": "target",
        "display_name": "Heart Disease",
        "positive_label": "Heart disease present",
        "negative_label": "No heart disease",
        "features": {
            "age":      {"label": "Age",                              "type": "number", "min": 1,   "max": 120, "default": 50},
            "sex":      {"label": "Sex",                               "type": "select", "options": {"Male": 1, "Female": 0}},
            "cp":       {"label": "Chest Pain Type (0-3)",             "type": "number", "min": 0,   "max": 3,   "default": 0},
            "trestbps": {"label": "Resting Blood Pressure (mm Hg)",    "type": "number", "min": 80,  "max": 220, "default": 120},
            "chol":     {"label": "Serum Cholesterol (mg/dl)",         "type": "number", "min": 100, "max": 600, "default": 200},
            "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl",   "type": "select", "options": {"Yes": 1, "No": 0}},
            "restecg":  {"label": "Resting ECG Result (0-2)",          "type": "number", "min": 0,   "max": 2,   "default": 0},
            "thalach":  {"label": "Max Heart Rate Achieved",           "type": "number", "min": 60,  "max": 220, "default": 150},
            "exang":    {"label": "Exercise Induced Angina",           "type": "select", "options": {"Yes": 1, "No": 0}},
            "oldpeak":  {"label": "ST Depression (exercise vs rest)",  "type": "number", "min": 0.0, "max": 7.0, "default": 1.0, "step": 0.1},
            "slope":    {"label": "Slope of Peak Exercise ST (0-2)",   "type": "number", "min": 0,   "max": 2,   "default": 1},
            "ca":       {"label": "Number of Major Vessels (0-4)",     "type": "number", "min": 0,   "max": 4,   "default": 0},
            "thal":     {"label": "Thalassemia (1=normal,2=fixed,3=reversible)", "type": "number", "min": 0, "max": 3, "default": 2},
        },
    },
    "diabetes": {
        "csv": os.path.join(DATA_DIR, "diabetes.csv"),
        "target_col": "Outcome",
        "display_name": "Diabetes",
        "positive_label": "Diabetes likely",
        "negative_label": "Diabetes unlikely",
        "features": {
            "Pregnancies":              {"label": "Number of Pregnancies",        "type": "number", "min": 0,   "max": 20,  "default": 1},
            "Glucose":                  {"label": "Glucose Level",                "type": "number", "min": 0,   "max": 250, "default": 120},
            "BloodPressure":            {"label": "Blood Pressure (mm Hg)",       "type": "number", "min": 0,   "max": 150, "default": 70},
            "SkinThickness":            {"label": "Skin Thickness (mm)",          "type": "number", "min": 0,   "max": 100, "default": 20},
            "Insulin":                  {"label": "Insulin Level",                "type": "number", "min": 0,   "max": 900, "default": 80},
            "BMI":                      {"label": "BMI",                          "type": "number", "min": 0.0, "max": 70.0,"default": 25.0, "step": 0.1},
            "DiabetesPedigreeFunction": {"label": "Diabetes Pedigree Function",   "type": "number", "min": 0.0, "max": 3.0, "default": 0.5,  "step": 0.01},
            "Age":                      {"label": "Age",                         "type": "number", "min": 1,   "max": 120, "default": 35},
        },
    },
    "breast_cancer": {
        "csv": os.path.join(DATA_DIR, "breast_cancer.csv"),
        "target_col": "target",
        "display_name": "Breast Cancer",
        # NOTE: in sklearn's breast cancer dataset, 0 = malignant, 1 = benign
        "positive_label": "Benign (non-cancerous)",
        "negative_label": "Malignant (cancerous)",
        # 30 numeric features -- built dynamically from the CSV columns
        # rather than typed out by hand (see src/train.py), since a form
        # with 30 manual fields would be unmaintainable here.
        "features": "auto",
    },
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
