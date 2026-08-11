"""
Step 4: Recommendation Engine
-------------------------------
Scores every simulated scenario on a shipping-efficiency vs. profitability
composite, then recommends:
  1. Ship Mode changes  -- per (Factory, Region), the mode with the best
     composite score, when it differs from what's used today.
  2. Regional reallocation flags -- per Factory, the Regions performing in
     the bottom quartile on today's actual numbers, i.e. the strongest
     candidates for a dedicated regional fulfillment point.

Usage:
    python recommend.py --outdir out --w-leadtime 0.5 --w-margin 0.5
"""
import argparse
import os

import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)


def score_scenarios(scenarios: pd.DataFrame, w_leadtime: float, w_margin: float) -> pd.DataFrame:
    scenarios = scenarios.copy()
    # lower lead time is better -> negate its z-score; higher margin is better
    scenarios["Score"] = (
        -w_leadtime * zscore(scenarios["Predicted Lead Time Days"])
        + w_margin * zscore(scenarios["Predicted Margin Pct"])
    )
    return scenarios


def ship_mode_recommendations(scenarios: pd.DataFrame) -> pd.DataFrame:
    best = scenarios.loc[scenarios.groupby(["Factory", "Region"])["Score"].idxmax()].copy()
    best["Recommend Switch"] = ~best["Is Current Ship Mode"]
    cols = [
        "Factory",
        "Region",
        "Orders",
        "Dominant_Ship_Mode",
        "Ship Mode",
        "Recommend Switch",
        "Current_Lead_Time_Days",
        "Predicted Lead Time Days",
        "Delta Lead Time Days",
        "Current_Margin_Pct",
        "Predicted Margin Pct",
        "Delta Margin Pct",
        "Score",
    ]
    best = best[cols].rename(columns={"Ship Mode": "Recommended Ship Mode"})
    return best.sort_values("Score", ascending=False).reset_index(drop=True)


def regional_reallocation_flags(baseline: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline.copy()
    baseline["Lead Time Z"] = baseline.groupby("Factory")["Current_Lead_Time_Days"].transform(zscore)
    baseline["Margin Z"] = baseline.groupby("Factory")["Current_Margin_Pct"].transform(zscore)
    baseline["Performance Score"] = -baseline["Lead Time Z"] + baseline["Margin Z"]

    q1 = baseline.groupby("Factory")["Performance Score"].transform(lambda s: s.quantile(0.25))
    baseline["Reallocation Candidate"] = baseline["Performance Score"] <= q1

    cols = [
        "Factory",
        "Region",
        "Orders",
        "Current_Lead_Time_Days",
        "Current_Margin_Pct",
        "Performance Score",
        "Reallocation Candidate",
    ]
    return baseline[cols].sort_values(["Factory", "Performance Score"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--w-leadtime", type=float, default=0.5)
    ap.add_argument("--w-margin", type=float, default=0.5)
    args = ap.parse_args()

    scenarios = pd.read_csv(os.path.join(args.outdir, "scenario_results.csv"))
    baseline = pd.read_csv(os.path.join(args.outdir, "baseline_by_factory_region.csv"))

    scenarios = score_scenarios(scenarios, args.w_leadtime, args.w_margin)

    ship_mode_recs = ship_mode_recommendations(scenarios)
    reallocation_flags = regional_reallocation_flags(baseline)

    ship_mode_path = os.path.join(args.outdir, "recommendations_ship_mode.csv")
    reallocation_path = os.path.join(args.outdir, "recommendations_reallocation.csv")
    ship_mode_recs.to_csv(ship_mode_path, index=False)
    reallocation_flags.to_csv(reallocation_path, index=False)

    print(f"Wrote {ship_mode_path}")
    print(f"Wrote {reallocation_path}\n")

    print("=== Ship Mode recommendations (top 10 by score) ===")
    print(ship_mode_recs.head(10).to_string(index=False))

    print("\n=== Regions flagged for reallocation (bottom quartile per Factory) ===")
    print(reallocation_flags[reallocation_flags["Reallocation Candidate"]].to_string(index=False))


if __name__ == "__main__":
    main()
