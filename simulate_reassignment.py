"""
Step: Scenario Simulation Engine (Factory Reassignment)
-----------------------------------------------------------
Complements simulate.py (which varies Ship Mode within a product's current
factory) by varying the FACTORY itself. For every (Product, Region) combo
observed in the data, every candidate factory -- including the one already
serving it -- is scored with the trained Lead Time / Margin models, using
that candidate factory's typical (dominant) Ship Mode for the region as its
operating profile. Product economics (units, price, cost) are held constant
since they belong to the product, not the plant producing it.

This answers: "if this product's Region-X demand were served by Factory Y
instead of Factory X, what would lead time and margin look like, and by how
much would that change operational and profit outcomes?"

Usage:
    python simulate_reassignment.py --outdir out
"""
import argparse
import os

import joblib
import json
import pandas as pd

from data_prep import ONE_HOT_COLS, SHIP_SPEED_RANK
from simulate import align_to_feature_columns


def build_product_region_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["Factory", "Region", "Product Name"])
        .agg(
            Orders=("Row ID", "count"),
            Current_Lead_Time_Days=("Lead Time Days", "mean"),
            Current_Margin_Pct=("Margin Pct", "mean"),
            Units=("Units", "median"),
            Sales_per_Unit=("Sales per Unit", "median"),
            Cost_per_Unit=("Cost per Unit", "median"),
            State_Freq=("State/Province Freq", "median"),
            City_Freq=("City Freq", "median"),
            Dominant_Country=("Country/Region", lambda s: s.mode().iat[0]),
        )
        .reset_index()
        .rename(columns={"Factory": "Home_Factory"})
    )
    return table


def factory_region_ship_mode_lookup(baseline: pd.DataFrame) -> dict:
    """(Factory, Region) -> Dominant_Ship_Mode, with a per-Factory fallback."""
    lookup = {
        (row["Factory"], row["Region"]): row["Dominant_Ship_Mode"] for _, row in baseline.iterrows()
    }
    factory_fallback = {
        factory: g.loc[g["Orders"].idxmax(), "Dominant_Ship_Mode"]
        for factory, g in baseline.groupby("Factory")
    }
    return lookup, factory_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.outdir, "processed.csv"))
    baseline = pd.read_csv(os.path.join(args.outdir, "baseline_by_factory_region.csv"))
    with open(os.path.join(args.outdir, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    leadtime_model = joblib.load(os.path.join(args.outdir, "model_lead_time_days.pkl"))
    margin_model = joblib.load(os.path.join(args.outdir, "model_margin_pct.pkl"))

    product_region = build_product_region_table(df)
    factories = sorted(df["Factory"].unique())
    ship_mode_lookup, ship_mode_fallback = factory_region_ship_mode_lookup(baseline)

    rows = []
    for _, pr in product_region.iterrows():
        for candidate in factories:
            ship_mode = ship_mode_lookup.get((candidate, pr["Region"]), ship_mode_fallback[candidate])
            rows.append(
                {
                    "Product Name": pr["Product Name"],
                    "Region": pr["Region"],
                    "Home Factory": pr["Home_Factory"],
                    "Candidate Factory": candidate,
                    "Is Home Factory": candidate == pr["Home_Factory"],
                    "Orders": pr["Orders"],
                    "Current_Lead_Time_Days": pr["Current_Lead_Time_Days"],
                    "Current_Margin_Pct": pr["Current_Margin_Pct"],
                    # model-input columns (named to match the training feature matrix)
                    "Factory": candidate,
                    "Ship Mode": ship_mode,
                    "Country/Region": pr["Dominant_Country"],
                    "Units": pr["Units"],
                    "Sales per Unit": pr["Sales_per_Unit"],
                    "Cost per Unit": pr["Cost_per_Unit"],
                    "Ship Speed Rank": SHIP_SPEED_RANK[ship_mode],
                    "State/Province Freq": pr["State_Freq"],
                    "City Freq": pr["City_Freq"],
                }
            )
    scenarios = pd.DataFrame(rows)

    X_scenarios = align_to_feature_columns(scenarios, feature_cols)
    scenarios["Predicted Lead Time Days"] = leadtime_model.predict(X_scenarios)
    scenarios["Predicted Margin Pct"] = margin_model.predict(X_scenarios)

    scenarios["Delta Lead Time Days"] = (
        scenarios["Predicted Lead Time Days"] - scenarios["Current_Lead_Time_Days"]
    )
    scenarios["Delta Margin Pct"] = scenarios["Predicted Margin Pct"] - scenarios["Current_Margin_Pct"]

    keep_cols = [
        "Product Name",
        "Region",
        "Home Factory",
        "Candidate Factory",
        "Is Home Factory",
        "Orders",
        "Units",
        "Sales per Unit",
        "Current_Lead_Time_Days",
        "Predicted Lead Time Days",
        "Delta Lead Time Days",
        "Current_Margin_Pct",
        "Predicted Margin Pct",
        "Delta Margin Pct",
    ]
    scenarios = scenarios[keep_cols]

    out_path = os.path.join(args.outdir, "scenario_factory_reassignment.csv")
    scenarios.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({scenarios.shape})")


if __name__ == "__main__":
    main()
