# Insurance Claim Cost Prediction (Regression)

Predicting an individual's annual medical insurance charges from demographic, lifestyle, and health attributes — so an insurer can estimate expected claim cost per policyholder.

## Problem

Accurately estimating expected medical costs per policyholder is central to insurance pricing and risk assessment. This project builds a regression model that predicts continuous dollar charges from a handful of easily collected attributes (age, sex, BMI, number of children, smoking status, region).

## Dataset

[Medical Cost Personal Datasets](https://github.com/stedy/Machine-Learning-with-R-datasets) (`insurance.csv`, from Brett Lantz's *Machine Learning with R*) — 1,338 individuals, 7 columns, continuous target `charges`.

The raw CSV is **not committed to this repo**. The notebook downloads it automatically at runtime (with a fallback mirror if the primary source is unreachable) and caches it locally under `data/`, which is git-ignored. This keeps the repo lightweight and ensures anyone who clones it gets a working pipeline with zero manual data-download steps.

## Approach

1. **Data cleaning** — check for missing values and duplicate rows (a known minor quirk of this public dataset).
2. **EDA** — distribution of charges (raw and log-transformed), charges by smoking status, charges vs. BMI/age colored by smoking status, correlation heatmap.
3. **Preprocessing** — `ColumnTransformer` (standard scaling for numeric features, one-hot encoding for categorical features) wrapped in an sklearn `Pipeline`, so there's no train/test leakage.
4. **Modeling** — two models trained and compared:
   - **Linear Regression** — interpretable baseline (coefficients are directly readable as dollar effects)
   - **XGBoost Regressor** — captures non-linear interactions (e.g. BMI × smoking status) a linear model misses
5. **Evaluation** — RMSE, MAE, R², predicted-vs-actual scatter plots, and residual distributions for both models.
6. **Feature importance** — top drivers of cost according to XGBoost, plus the raw Linear Regression coefficients for direct dollar interpretation.

## Results

See the notebook's Section 7 (`results_df`) for the full metrics table generated on each run — this file intentionally doesn't hardcode numbers here so the README never drifts out of sync with the code.

## Reproducibility

- Fixed random seed (`RANDOM_SEED = 42`) used for the train/test split and both models.
- Dependency versions pinned in `requirements.txt`.
- No absolute file paths — all paths are relative to the repo root.
- Data loading is a documented function (`load_insurance_data`) with an explicit fallback source, rather than a manual download step.

Re-running the notebook top to bottom reproduces the same split, models, and metrics.

## How to run

**Option A — Google Colab (recommended, no local setup):**
1. Upload `Insurance_Claim_Cost_Prediction.ipynb` to [Google Colab](https://colab.research.google.com/).
2. In the first code cell, install dependencies: `!pip install -r requirements.txt` (or just run the notebook — Colab has most of these pre-installed; only `xgboost` may need `!pip install xgboost`).
3. Run all cells (`Runtime` → `Run all`). Data downloads automatically.

**Option B — Local:**
```bash
git clone https://github.com/<your-username>/insurance-claim-cost-regression.git
cd insurance-claim-cost-regression
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Insurance_Claim_Cost_Prediction.ipynb
```

## Repository structure

```
insurance-claim-cost-regression/
├── Insurance_Claim_Cost_Prediction.ipynb   # Full analysis, pipeline, models, evaluation
├── requirements.txt                        # Pinned dependencies
└── README.md
```

## Possible extensions

- Hyperparameter tuning (`GridSearchCV`, `Optuna`)
- Explicit interaction features (e.g. `bmi * smoker`) for the linear model
- SHAP values for per-policyholder explanations
- Modeling `log(charges)` directly and comparing back-transformed metrics
