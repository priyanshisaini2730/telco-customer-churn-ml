"""
batch_scoring.py — Business Intelligence Layer for Telco Churn Project

PURPOSE
-------
This script is the "beyond accuracy" deliverable of the project. It takes
already-scored customers (i.e. each customer already has a
predicted_churn_probability from the trained model) and turns that raw
probability into something a business can act on:

    risk tier  +  value tier  ->  recommended retention action
                              ->  aggregate revenue / CLV-at-stake numbers

DESIGN DECISIONS (documented here so they're easy to explain in an interview)
------------------------------------------------------------------------------
1. Risk tier and Value tier are INDEPENDENT axes.
   - Risk tier  = tercile split on predicted_churn_probability
   - Value tier = tercile split on MonthlyCharges (NOT on CLV)

   Why not CLV as the second axis? Because CLV is mathematically DERIVED
   from churn probability (CLV = MonthlyCharges x min(1/prob, cap)). Using
   CLV against Risk would create a built-in negative correlation -- a
   customer can't easily be both "High Risk" and "High CLV" by construction,
   since high churn probability mechanically shrinks the CLV multiplier.
   MonthlyCharges is risk-independent, so pairing it with risk tier gives a
   genuinely orthogonal 3x3 matrix where every cell is reachable.

2. CLV is still computed and reported (per customer AND per segment) as the
   "revenue at stake" headline metric -- it's just not used to CUT the tiers.

3. Tercile tiers (qcut into 3 equal-sized groups) are used instead of fixed
   thresholds (e.g. "prob > 0.6 = High"). This is defensible in an interview:
   "High Risk = the riskiest third of OUR customer base" needs no arbitrary
   cutoff justification, whereas a fixed number always invites "why that
   number and not another?"

4. Architectural note: this script assumes prediction (model inference) has
   ALREADY happened upstream and produced predicted_churn_probability per
   customer. Scoring/inference logic and business-segmentation logic are
   kept separate (single responsibility) -- same principle as keeping
   add_rfm_features() shared between training and serving instead of
   duplicating transform logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input: a CSV that already has predicted_churn_probability per customer.
# (Produced by a batch inference step -- see note in main() if this doesn't
#  exist yet.)
SCORED_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "scored_customers.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PER_CUSTOMER_PATH = OUTPUT_DIR / "customer_action_scores.csv"
OUTPUT_SEGMENT_SUMMARY_PATH = OUTPUT_DIR / "segment_summary.csv"

CLV_CAP_MONTHS = 60  # matches the locked CLV formula from the project

RISK_TIER_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
VALUE_TIER_LABELS = ["Low Revenue", "Medium Revenue", "High Revenue"]

# The 3x3 action matrix agreed on for this project.
ACTION_MATRIX = {
    ("High Risk", "High Revenue"): "Aggressive retention (personal call + custom discount)",
    ("High Risk", "Medium Revenue"): "Targeted automated offer (email/SMS discount)",
    ("High Risk", "Low Revenue"): "Low-cost nudge (email only)",
    ("Medium Risk", "High Revenue"): "Proactive loyalty perk (early feature access)",
    ("Medium Risk", "Medium Revenue"): "Standard engagement (monitor, no action)",
    ("Medium Risk", "Low Revenue"): "Minimal intervention",
    ("Low Risk", "High Revenue"): "Upsell candidate (premium tier offer)",
    ("Low Risk", "Medium Revenue"): "Standard service (no action)",
    ("Low Risk", "Low Revenue"): "Deprioritize (no action)",
}


def compute_clv(monthly_charges: pd.Series, churn_probability: pd.Series,
                 cap_months: int = CLV_CAP_MONTHS) -> pd.Series:
    """
    CLV = MonthlyCharges x min(1 / predicted_churn_probability, cap_months)

    Guards against churn_probability == 0, which would otherwise cause a
    divide-by-zero -> inf. A probability of exactly 0 is treated as "as
    retained as the model allows", i.e. capped at cap_months.
    """
    safe_prob = churn_probability.replace(0, np.nan)
    expected_months = (1.0 / safe_prob).clip(upper=cap_months)
    expected_months = expected_months.fillna(cap_months)
    return monthly_charges * expected_months


def tercile_tier(series: pd.Series, labels: list) -> pd.Series:
    """
    Splits a numeric series into 3 equal-sized groups (terciles) using
    pandas qcut: labels[0] = bottom third, labels[1] = middle third,
    labels[2] = top third.

    duplicates="drop" guards against tied bin edges on skewed/discrete data
    (e.g. many customers sharing the exact same MonthlyCharges value) -- in
    that case pandas falls back to fewer, unevenly-sized bins rather than
    raising an error.

    Cast to plain string (instead of leaving it as pandas Categorical):
    Categorical columns carry their OWN internal category order, which can
    silently override intended sort order later (e.g. sort_values on a
    Categorical sorts by category order, not by a mapped numeric rank) --
    plain strings avoid that trap.
    """
    return pd.qcut(series, q=3, labels=labels, duplicates="drop").astype(str)


def assign_action(row: pd.Series) -> str:
    key = (row["risk_tier"], row["value_tier"])
    return ACTION_MATRIX.get(key, "Review manually (tier combination not mapped)")


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df with at least:
      - MonthlyCharges
      - predicted_churn_probability   (model's predict_proba output)

    Adds: CLV, risk_tier, value_tier, recommended_action
    """
    df = df.copy()

    df["CLV"] = compute_clv(df["MonthlyCharges"], df["predicted_churn_probability"])
    df["risk_tier"] = tercile_tier(df["predicted_churn_probability"], RISK_TIER_LABELS)
    df["value_tier"] = tercile_tier(df["MonthlyCharges"], VALUE_TIER_LABELS)
    df["recommended_action"] = df.apply(assign_action, axis=1)

    return df


def build_segment_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per (risk_tier, value_tier) segment:
      - customer_count
      - total_monthly_revenue   : sum of MonthlyCharges in the segment
      - total_clv_at_stake      : sum of CLV in the segment (revenue lost
                                   if every customer in this segment churns)

    NOTE: An "estimated revenue saved" figure (CLV-at-stake x an assumed
    retention-campaign success rate) was deliberately left OUT of this
    summary. This dataset is a churn snapshot -- no retention campaign was
    ever run on these customers, so there's no intervention-outcome data to
    ground a success rate in. CLV-at-stake is reported because it IS
    statistically grounded (built from the model's own predicted
    probabilities); a savings figure built on an unmeasured success rate
    would not be, so it's excluded rather than presented as if it were.
    """
    summary = (
        scored_df.groupby(["risk_tier", "value_tier"], observed=True)
        .agg(
            customer_count=("CLV", "size"),
            total_monthly_revenue=("MonthlyCharges", "sum"),
            total_clv_at_stake=("CLV", "sum"),
        )
        .reset_index()
    )

    summary["recommended_action"] = summary.apply(assign_action, axis=1)

    # Order rows High -> Medium -> Low on both axes for a readable report
    tier_order = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
    value_order = {"High Revenue": 0, "Medium Revenue": 1, "Low Revenue": 2}
    summary["_r"] = summary["risk_tier"].map(tier_order)
    summary["_v"] = summary["value_tier"].map(value_order)
    summary = summary.sort_values(["_r", "_v"]).drop(columns=["_r", "_v"]).reset_index(drop=True)

    return summary


def print_headline_numbers(summary_df: pd.DataFrame) -> None:
    """
    Prints the interview-ready headline numbers, e.g.:
    "High Risk + High Revenue segment: 412 customers, Rs X total monthly
    revenue at stake, Rs Y total CLV at stake."

    Deliberately stops at CLV-at-stake and does NOT extrapolate a "money
    saved" figure -- that would require a retention-campaign success rate,
    which this dataset has no way to ground (see build_segment_summary
    docstring).
    """
    hr_hv = summary_df[
        (summary_df["risk_tier"] == "High Risk") & (summary_df["value_tier"] == "High Revenue")
    ]

    total_high_risk_clv = summary_df.loc[summary_df["risk_tier"] == "High Risk", "total_clv_at_stake"].sum()

    print("=" * 70)
    print("BUSINESS IMPACT SUMMARY")
    print("=" * 70)

    if not hr_hv.empty:
        row = hr_hv.iloc[0]
        print(f"\nHigh Risk + High Revenue segment: {int(row['customer_count'])} customers")
        print(f"  -> Monthly revenue at stake: Rs {row['total_monthly_revenue']:,.0f}")
        print(f"  -> Total CLV at stake (if all churn): Rs {row['total_clv_at_stake']:,.0f}")

    print(f"\nAcross ALL High Risk customers:")
    print(f"  -> Total CLV at stake: Rs {total_high_risk_clv:,.0f}")
    print(
        "\n(Note: this is CLV-at-stake, i.e. revenue lost if every High Risk "
        "customer churns -- not an estimated 'savings' figure. A savings "
        "estimate would require a retention-campaign success rate, which "
        "this dataset has no intervention-outcome data to ground.)"
    )
    print("=" * 70)


def main():
    print(f"Loading scored customer data from: {SCORED_INPUT_PATH}")

    if not SCORED_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"'{SCORED_INPUT_PATH}' not found.\n"
            f"This script expects a CSV that already has a "
            f"'predicted_churn_probability' column per customer (i.e. batch "
            f"inference has already run). If that step doesn't exist yet in "
            f"the pipeline, it needs to be added before this script, or "
            f"this script's data-loading section needs to call the model "
            f"directly -- worth confirming which path fits the project before "
            f"running this."
        )

    df = pd.read_csv(SCORED_INPUT_PATH)

    required_cols = {"MonthlyCharges", "predicted_churn_probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required column(s): {missing}. "
            f"Run the inference/prediction step first to generate "
            f"'predicted_churn_probability' before running batch scoring."
        )

    scored_df = score_customers(df)
    segment_summary = build_segment_summary(scored_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(OUTPUT_PER_CUSTOMER_PATH, index=False)
    segment_summary.to_csv(OUTPUT_SEGMENT_SUMMARY_PATH, index=False)

    print(f"\nPer-customer scores saved to: {OUTPUT_PER_CUSTOMER_PATH}")
    print(f"Segment summary saved to: {OUTPUT_SEGMENT_SUMMARY_PATH}\n")

    print_headline_numbers(segment_summary)


if __name__ == "__main__":
    main()
