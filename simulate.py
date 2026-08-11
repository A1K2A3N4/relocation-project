"""
Step 3: Scenario Simulation
-----------------------------
For every (Factory, Region) combination, builds synthetic "what-if" order
profiles for each candidate Ship Mode, scores them with the trained models,
and compares the predicted outcome against the actual current baseline.

This answers: "if Factory X's orders into Region Y shipped via mode Z
instead of what's used today, what would lead time and margin look like?"

Usage:
    python simulate.py --outdir out
"""
import argparse
import itertools
import json
import os

import joblib
import pandas as pd

from data_prep import ONE_HOT_COLS, SHIP_SPEED_RANK


def build_scenario_row(factory, region, ship_mode, country, units, sales_pu, cost_pu, state_freq, city_freq):
    return {
        "Factory": factory,
        "Region": region,
        "Ship Mode": ship_mode,
        "Country/Region": country,
        "Units": units,
        "Sales per Unit": sales_pu,
        "Cost per Unit": cost_pu,
        "Ship Speed Rank": SHIP_SPEED_RANK[ship_mode],
        "State/Province Freq": state_freq,
        "City Freq": city_freq,
    }


def align_to_feature_columns(raw_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    dummies = pd.get_dummies(raw_df[ONE_HOT_COLS], prefix=ONE_HOT_COLS)
    numeric_cols = [
        "Units",
        "Sales per Unit",
        "Cost per Unit",
        "Ship Speed Rank",
        "State/Province Freq",
        "City Freq",
    ]
    X = pd.concat([raw_df[numeric_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    # scenarios can't produce every one-hot level (e.g. Country) that training saw;
    # reindex to the training columns and zero-fill anything missing
    X = X.reindex(columns=feature_cols, fill_value=0)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    baseline = pd.read_csv(os.path.join(args.outdir, "baseline_by_factory_region.csv"))
    with open(os.path.join(args.outdir, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    leadtime_model = joblib.load(os.path.join(args.outdir, "model_lead_time_days.pkl"))
    margin_model = joblib.load(os.path.join(args.outdir, "model_margin_pct.pkl"))

    ship_modes = list(SHIP_SPEED_RANK.keys())
    rows = []
    for _, b in baseline.iterrows():
        for ship_mode in ship_modes:
            rows.append(
                build_scenario_row(
                    factory=b["Factory"],
                    region=b["Region"],
                    ship_mode=ship_mode,
                    country=b["Dominant_Country"],
                    units=b["Units"],
                    sales_pu=b["Sales_per_Unit"],
                    cost_pu=b["Cost_per_Unit"],
                    state_freq=b["State_Freq"],
                    city_freq=b["City_Freq"],
                )
            )
    scenarios = pd.DataFrame(rows)

    X_scenarios = align_to_feature_columns(scenarios, feature_cols)
    scenarios["Predicted Lead Time Days"] = leadtime_model.predict(X_scenarios)
    scenarios["Predicted Margin Pct"] = margin_model.predict(X_scenarios)

    scenarios = scenarios.merge(
        baseline[
            [
                "Factory",
                "Region",
                "Orders",
                "Current_Lead_Time_Days",
                "Current_Margin_Pct",
                "Dominant_Ship_Mode",
            ]
        ],
        on=["Factory", "Region"],
        how="left",
    )
    scenarios["Is Current Ship Mode"] = scenarios["Ship Mode"] == scenarios["Dominant_Ship_Mode"]
    scenarios["Delta Lead Time Days"] = (
        scenarios["Predicted Lead Time Days"] - scenarios["Current_Lead_Time_Days"]
    )
    scenarios["Delta Margin Pct"] = scenarios["Predicted Margin Pct"] - scenarios["Current_Margin_Pct"]

    out_path = os.path.join(args.outdir, "scenario_results.csv")
    scenarios.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({scenarios.shape})")


if __name__ == "__main__":
    main()
