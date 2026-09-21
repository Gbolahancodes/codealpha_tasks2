# Disease Prediction from Medical Data

Predicts the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer**
from structured patient data using classical machine learning classifiers —
Logistic Regression, SVM, Random Forest, and XGBoost — with a Streamlit web
app for live predictions.

For each disease, all four algorithms are trained and compared automatically,
and the best-performing one (by F1 score) is saved and used for predictions.

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

Pick a disease, fill in patient data, get an instant prediction with a
confidence score.

---

## Results

| Dataset | Best Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Heart Disease | Random Forest | 0.82 | 0.85 | 0.91 |
| Diabetes | SVM | 0.75 | 0.63 | 0.79 |
| Breast Cancer | Logistic Regression | 0.98 | 0.99 | 1.00 |

Full per-algorithm metrics are in `models/*_metrics.json`.

---

## Datasets

| Dataset | Source | Rows | Features |
|---|---|---|---|
| Heart Disease | UCI Heart Disease (Cleveland) | 303 | 13 clinical features |
| Diabetes | Pima Indians Diabetes | 768 | 8 features |
| Breast Cancer | Wisconsin Diagnostic Breast Cancer (UCI, via scikit-learn) | 569 | 30 numeric features |

All three CSVs are included in `data/` — no download step required.

---

## Project Structure

```
disease-prediction/
├── data/                   # Datasets (CSV)
├── src/
│   ├── config.py           # Dataset + feature definitions
│   ├── data_utils.py       # Loading, splitting, scaling
│   ├── train.py            # Trains + compares all 4 models, saves the best
│   └── predict.py          # Loads a saved model, runs a prediction
├── models/                 # Pre-trained models + metrics (included)
├── app/
│   └── streamlit_app.py    # Streamlit web app
├── check_setup.py          # Verifies data/models are in place
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify setup

```bash
python check_setup.py
```

Datasets and trained models are already included, so you can skip straight
to step 3 — retraining is optional.

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Retraining

```bash
python -m src.train --dataset all        # all three datasets
python -m src.train --dataset heart      # or just one
python -m src.train --dataset diabetes
python -m src.train --dataset breast_cancer
```

Each run trains and scores all four algorithms, picks the best by F1,
and overwrites `models/<dataset>_model.joblib` and `_metrics.json`.

---

## Tech Stack

- **scikit-learn** — Logistic Regression, SVM, Random Forest, preprocessing
- **XGBoost** — gradient boosting classifier
- **pandas / numpy** — data handling
- **Streamlit** — web app
- **joblib** — model persistence

---

## Deployment

Deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repo.
3. Set the main file path to `app/streamlit_app.py`.
4. Deploy.

---

## Disclaimer

This project is for **educational purposes only**. Predictions are
statistical estimates based on patterns in public datasets — they are
**not medical diagnoses** and must not be used for real healthcare
decisions. Always consult a qualified healthcare professional.

---

## License

MIT — free to use, modify, and build on.
