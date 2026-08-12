"""
Step: Key Performance Indicators
------------------------------------
Summarizes the factory-reassignment recommendations into three KPIs:

  Lead Time Reduction (%)   Operational gain -- already computed per
                             recommendation by recommend_reassignment.py;
                             this step averages it across recommended moves.
  Profit Impact Stability   Financial safety -- how consistent the candidate
                             factory's margin has historically been (inverse
                             of its margin coefficient of variation). A
                             factory with volatile historical margins makes
                             any predicted profit gain less trustworthy.
  Scenario Confidence Score How much to trust a given simulated scenario --
                             combines the trained models' overall R2 (are the
                             Lead Time / Margin models any good?) with how
                             much historical data backs that specific
                             (Product, Region) combo (a recommendation drawn
                             from 5 orders is less trustworthy than one drawn
                             from 500).

Usage:
    python kpis.py --outdir out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def profit_impact_stability(recommendations: pd.DataFrame, risk: pd.DataFrame) -> pd.Series:
    margin_cv = risk.set_index("Factory")["Margin_CV"]
    candidate_cv = recommendations["Candidate Factory"].map(margin_cv)
    return 1.0 / (1.0 + candidate_cv.abs())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    recommendations = pd.read_csv(os.path.join(args.outdir, "recommendations_factory_reassignment.csv"))
    scenarios = pd.read_csv(os.path.join(args.outdir, "scenario_factory_reassignment.csv"))
    risk = pd.read_csv(os.path.join(args.outdir, "factory_risk_profile.csv"))
    with open(os.path.join(args.outdir, "model_metrics.json")) as f:
        model_metrics = json.load(f)

    orders_lookup = scenarios.set_index(["Product Name", "Region"])["Orders"].to_dict()
    recommendations["Orders"] = [
        orders_lookup[(p, r)] for p, r in zip(recommendations["Product Name"], recommendations["Region"])
    ]

    recommendations["Profit Impact Stability"] = profit_impact_stability(recommendations, risk)

    r2_leadtime = float(np.clip(model_metrics["Lead Time Days"]["r2"], 0, 1))
    r2_margin = float(np.clip(model_metrics["Margin Pct"]["r2"], 0, 1))
    model_quality = np.sqrt(r2_leadtime * r2_margin)

    orders_cap = recommendations["Orders"].max()
    data_sufficiency = recommendations["Orders"].apply(lambda o: min(1.0, np.log1p(o) / np.log1p(orders_cap)))
    recommendations["Scenario Confidence Score"] = model_quality * data_sufficiency

    detail_path = os.path.join(args.outdir, "kpi_detail.csv")
    recommendations.to_csv(detail_path, index=False)
    print(f"Wrote {detail_path}")

    recommended = recommendations[recommendations["Recommend Reassignment"]]
    summary = {
        "model_quality_leadtime_r2": r2_leadtime,
        "model_quality_margin_r2": r2_margin,
        "recommended_reassignments": int(len(recommended)),
        "avg_lead_time_reduction_pct": float(recommended["Lead Time Reduction (%)"].mean()) if len(recommended) else None,
        "avg_profit_impact_stability": float(recommended["Profit Impact Stability"].mean()) if len(recommended) else None,
        "avg_scenario_confidence_score": float(recommended["Scenario Confidence Score"].mean()) if len(recommended) else None,
        "total_estimated_profit_impact": float(recommended["Profit Impact ($)"].sum()) if len(recommended) else None,
    }

    summary_path = os.path.join(args.outdir, "kpi_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}\n")

    print("=== KPI Summary (recommended reassignments only) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Per-recommendation KPI detail (top 10 by Scenario Confidence Score) ===")
    print(
        recommended.sort_values("Scenario Confidence Score", ascending=False)
        .head(10)[
            [
                "Product Name",
                "Region",
                "Home Factory",
                "Candidate Factory",
                "Lead Time Reduction (%)",
                "Profit Impact Stability",
                "Scenario Confidence Score",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
