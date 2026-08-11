"""
Step 1: Data Preparation & Encoding
------------------------------------
Loads the raw Nassau Candy order-level CSV, engineers the fields the rest of
the pipeline needs (lead time, margin, a "Factory" proxy), removes extreme
outliers, and encodes categoricals into a model-ready feature matrix.

Factory proxy: the raw data has no Factory/plant column, so each product
Division (Chocolate / Sugar / Other) is treated as its own production line
("factory"). Reallocation decisions downstream are about which Region a
Division's factory should prioritize / how it should ship there — not about
moving a product to a different Division.

Usage:
    python data_prep.py --input "../Nassau Candy Distributor.csv" --outdir out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SHIP_SPEED_RANK = {
    "Same Day": 0,
    "First Class": 1,
    "Second Class": 2,
    "Standard Class": 3,
}

OUTLIER_COLS = ["Sales", "Cost", "Gross Profit", "Units", "Lead Time Days"]
ONE_HOT_COLS = ["Ship Mode", "Region", "Factory", "Country/Region"]
FREQ_ENCODE_COLS = ["State/Province", "City"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Factory"] = df["Division"]  # factory proxy: one production line per division

    df["Lead Time Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    # a shipment can't leave before it's ordered; drop the handful of rows where
    # that happens (data entry issues) rather than silently keeping bad labels
    df = df[df["Lead Time Days"] >= 0]

    df["Margin Pct"] = np.where(df["Sales"] > 0, df["Gross Profit"] / df["Sales"], np.nan)
    df["Sales per Unit"] = df["Sales"] / df["Units"].replace(0, np.nan)
    df["Cost per Unit"] = df["Cost"] / df["Units"].replace(0, np.nan)
    df["Ship Speed Rank"] = df["Ship Mode"].map(SHIP_SPEED_RANK)

    df = df.dropna(subset=["Margin Pct", "Sales per Unit", "Cost per Unit"])
    return df


def remove_outliers(df: pd.DataFrame, cols=OUTLIER_COLS, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]


def frequency_encode(df: pd.DataFrame, cols=FREQ_ENCODE_COLS) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        freq = df[col].value_counts(normalize=True)
        df[f"{col} Freq"] = df[col].map(freq)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """One-hot encode low-cardinality categoricals, keep engineered numerics.

    Assumes df has already been through frequency_encode().
    """
    dummies = pd.get_dummies(df[ONE_HOT_COLS], prefix=ONE_HOT_COLS)

    numeric_cols = [
        "Units",
        "Sales per Unit",
        "Cost per Unit",
        "Ship Speed Rank",
        "State/Province Freq",
        "City Freq",
    ]
    X = pd.concat([df[numeric_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X, list(X.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../Nassau Candy Distributor.csv")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)
    df = engineer_features(df)

    before = len(df)
    df = remove_outliers(df)
    print(f"Outlier removal: {before} -> {len(df)} rows ({before - len(df)} dropped)")

    df = frequency_encode(df)
    X, feature_cols = build_feature_matrix(df)

    processed_path = os.path.join(args.outdir, "processed.csv")
    features_path = os.path.join(args.outdir, "features.csv")
    cols_path = os.path.join(args.outdir, "feature_columns.json")
    baseline_path = os.path.join(args.outdir, "baseline_by_factory_region.csv")

    df.reset_index(drop=True).to_csv(processed_path, index=False)
    X.to_csv(features_path, index=False)
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # per (Factory, Region) baseline used by simulate.py to build "what-if" rows
    # and as the actual current-performance reference point
    baseline = (
        df.groupby(["Factory", "Region"])
        .agg(
            Orders=("Row ID", "count"),
            Current_Lead_Time_Days=("Lead Time Days", "mean"),
            Current_Margin_Pct=("Margin Pct", "mean"),
            Units=("Units", "median"),
            Sales_per_Unit=("Sales per Unit", "median"),
            Cost_per_Unit=("Cost per Unit", "median"),
            State_Freq=("State/Province Freq", "median"),
            City_Freq=("City Freq", "median"),
            Dominant_Ship_Mode=("Ship Mode", lambda s: s.mode().iat[0]),
            Dominant_Country=("Country/Region", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    baseline.to_csv(baseline_path, index=False)

    print(f"Wrote {processed_path} ({df.shape})")
    print(f"Wrote {features_path} ({X.shape})")
    print(f"Wrote {cols_path}")
    print(f"Wrote {baseline_path} ({baseline.shape})")


if __name__ == "__main__":
    main()
