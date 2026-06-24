"""
predict_batch.py — Batch Inference: produces predicted_churn_probability
for every customer in the dataset.

WHY THIS SCRIPT EXISTS (separate from inference.py's predict())
-----------------------------------------------------------------
inference.py loads the model via mlflow.pyfunc.load_model(), which is the
right choice for production serving — flavor-agnostic, stable API. But
PyFuncModel only exposes .predict(), which returns a HARD LABEL (0 or 1),
not a probability. That's fine for a single live "is this customer likely
to churn?" API call.

batch_scoring.py (the business-intelligence layer) needs a CONTINUOUS
churn probability per customer, because tercile-based risk segmentation
(splitting customers into Low/Medium/High risk thirds) is meaningless on
a binary 0/1 value — there's nothing to rank within "0" or within "1".

So this script loads the SAME trained model artifact a second time, but
through its NATIVE MLflow flavor (XGBoost or sklearn) instead of pyfunc.
Native flavors expose .predict_proba(), which is what we need here.

This isn't a workaround — production serving and batch business-analytics
scoring legitimately need different things from the same model, so they
access it differently. Both ultimately point at the identical trained
artifact on disk.

TRAIN/SERVE CONSISTENCY
------------------------
Feature transformation logic (_serve_transform) and the trained feature
schema (FEATURE_COLS) are imported directly from src.serving.inference
rather than reimplemented here. This guarantees batch predictions and live
API predictions go through the exact same transform path — same principle
as add_rfm_features() being a single shared function for training and
serving instead of two copies that could drift apart.

NOTE: importing src.serving.inference also triggers its top-level pyfunc
model-load (you'll see its "Model loaded successfully" print). That's a
known, harmless side effect of reusing its code — it does mean the model
gets loaded into memory twice (once as pyfunc, once via native flavor
below). Fine for a one-off batch run; worth refactoring later (splitting
_serve_transform/FEATURE_COLS into their own module with no loading side
effect) if this script ends up running often.
"""

import os
import sys
import pandas as pd
import mlflow.xgboost
import mlflow.sklearn

# Make the project root importable regardless of which directory this
# script is launched from. `python scripts/predict_batch.py` only adds
# the scripts/ folder itself to sys.path, not the project root -- so
# `from src...` fails unless we explicitly add the parent folder here.
# Mirrors the same Windows-path-safety mindset used elsewhere in this
# project (the pathlib fix for the MLflow tracking URI).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.load_data import load_data
from src.serving.inference import _serve_transform, FEATURE_COLS, MODEL_DIR

RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/processed/scored_customers.csv"

# Columns present in the raw CSV that are NOT model features and must be
# dropped before _serve_transform — otherwise they get one-hot encoded
# as ordinary object columns (customerID alone would explode into
# thousands of dummy columns). The live API never receives these either.
NON_FEATURE_COLUMNS = ["customerID", "Churn"]


def load_probability_model(model_dir: str):
    """
    Loads the trained model via its NATIVE MLflow flavor (not pyfunc), so
    .predict_proba() is available. Tries XGBoost flavor first (this
    project's model type), falls back to sklearn flavor in case the model
    was logged with mlflow.sklearn.log_model() instead — both are valid
    ways to log an XGBoost sklearn-API model, so trying both costs little.
    """
    try:
        m = mlflow.xgboost.load_model(model_dir)
        print(f"✅ Loaded native XGBoost-flavor model from {model_dir}")
        return m
    except Exception as xgb_error:
        try:
            m = mlflow.sklearn.load_model(model_dir)
            print(f"✅ Loaded native sklearn-flavor model from {model_dir}")
            return m
        except Exception as sklearn_error:
            raise Exception(
                "Could not load the model in a flavor that exposes "
                "predict_proba.\n"
                f"  XGBoost flavor load failed: {xgb_error}\n"
                f"  sklearn flavor load failed: {sklearn_error}\n"
                "Check how the model was logged during training "
                "(mlflow.xgboost.log_model vs mlflow.sklearn.log_model) "
                "and confirm which flavor matches."
            )


def main():
    model = load_probability_model(MODEL_DIR)

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    raw_df = load_data(RAW_DATA_PATH)

    # Preserve these in their original, human-readable form for the
    # output file — _serve_transform will encode/drop/reindex columns,
    # so grab what batch_scoring.py needs BEFORE transforming.
    customer_ids = raw_df["customerID"] if "customerID" in raw_df.columns else pd.Series(range(len(raw_df)))
    monthly_charges = raw_df["MonthlyCharges"]

    transform_input = raw_df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")

    print("Applying serving-time feature transformation (same path as the live API)...")
    X = _serve_transform(transform_input)

    if list(X.columns) != FEATURE_COLS:
        # _serve_transform already reindexes to FEATURE_COLS internally,
        # so this should never trigger — but it's cheap insurance against
        # silent train/serve skew if that function ever changes.
        raise AssertionError(
            "Transformed columns don't match FEATURE_COLS after "
            "_serve_transform — investigate before trusting predictions."
        )

    print(f"Running batch prediction on {len(X)} customers...")
    churn_probabilities = model.predict_proba(X)[:, 1]

    scored_df = pd.DataFrame({
        "customerID": customer_ids.values,
        "MonthlyCharges": monthly_charges.values,
        "predicted_churn_probability": churn_probabilities,
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    scored_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved {len(scored_df)} scored customers to: {OUTPUT_PATH}")
    print(scored_df.head())


if __name__ == "__main__":
    main()
