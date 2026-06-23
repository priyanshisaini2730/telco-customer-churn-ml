import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
NO_SERVICE_VALUES = {"No", "No internet service", "No phone service"}


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 3 RFM-inspired features. Must run on RAW columns, before any
    binary/one-hot encoding — both training (build_features.py) and
    serving (inference.py) call this at that exact point in their pipelines.

    - num_services            (Frequency proxy)
    - charge_per_tenure_ratio (Monetary proxy)
    - tenure_bucket           (Recency proxy)
    """
    df = df.copy()

    # --- Frequency: num_services ---
    present_cols = [c for c in SERVICE_COLS if c in df.columns]
    df["num_services"] = df[present_cols].apply(
        lambda col: ~col.astype(str).isin(NO_SERVICE_VALUES)
    ).sum(axis=1)

    # --- Monetary: charge_per_tenure_ratio ---
    df["charge_per_tenure_ratio"] = df["TotalCharges"] / df["tenure"]
    zero_tenure = df["tenure"] == 0
    df.loc[zero_tenure, "charge_per_tenure_ratio"] = df.loc[zero_tenure, "MonthlyCharges"]

    df = df.drop(columns=["TotalCharges"])

    # --- Recency: tenure_bucket ---
    # .astype(str) matters: pd.cut returns a Categorical dtype, and
    # build_features.py only looks for dtype == "object" to decide what
    # gets one-hot encoded. Without this cast, tenure_bucket would silently
    # skip encoding entirely.
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Established", "Loyal", "Veteran"],
    ).astype(str)

    return df