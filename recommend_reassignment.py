"""
Step: Optimization & Recommendation Logic (Factory Reassignment)
----------------------------------------------------------------
Ranks candidate factories for each (Product, Region) combo produced by
simulate_reassignment.py on three axes:
  1. Lead Time Reduction (%) -- operational gain vs. the current factory
  2. Risk Reduction          -- change in operational consistency, proxied
                                 by each factory's historical lead-time
                                 coefficient of variation (CV) across all the
                                 regions it already serves: a factory whose
                                 lead times are historically more consistent
                                 is lower-risk to route more volume through
  3. Profit Impact ($)       -- Delta Margin Pct applied to the product's
                                 estimated demand volume (Orders x Units x
                                 Sales per Unit)

The three axes are z-scored and combined into a composite score; for each
(Product, Region) the top-N candidate factories that beat the current
factory are kept as reassignment recommendations.

Usage:
    python recommend_reassignment.py --outdir out --top-n 2 \
        --w-leadtime 0.4 --w-risk 0.3 --w-profit 0.3
"""
import argparse
import os

import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)


def factory_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-factory historical CV of lead time & margin -- higher CV = more volatile/risky."""
    g = df.groupby("Factory")["Lead Time Days"].agg(["mean", "std"])
    lead_cv = (g["std"] / g["mean"]).rename("Lead_Time_CV")
    g2 = df.groupby("Factory")["Margin Pct"].agg(["mean", "std"])
    margin_cv = (g2["std"] / g2["mean"].abs()).rename("Margin_CV")
    return pd.concat([lead_cv, margin_cv], axis=1).reset_index()


def score_candidates(scenarios: pd.DataFrame, risk: pd.DataFrame, w_leadtime, w_risk, w_profit) -> pd.DataFrame:
    df = scenarios.merge(
        risk.rename(columns={"Factory": "Home Factory", "Lead_Time_CV": "Home_Lead_CV"})[["Home Factory", "Home_Lead_CV"]],
        on="Home Factory",
        how="left",
    ).merge(
        risk.rename(columns={"Factory": "Candidate Factory", "Lead_Time_CV": "Candidate_Lead_CV"})[
            ["Candidate Factory", "Candidate_Lead_CV"]
        ],
        on="Candidate Factory",
        how="left",
    )

    df["Lead Time Reduction (%)"] = (
        -df["Delta Lead Time Days"] / df["Current_Lead_Time_Days"] * 100
    )
    df["Risk Reduction"] = df["Home_Lead_CV"] - df["Candidate_Lead_CV"]

    estimated_demand = df["Orders"] * df["Units"] * df["Sales per Unit"]
    df["Profit Impact ($)"] = df["Delta Margin Pct"] * estimated_demand

    df["Composite Score"] = (
        w_leadtime * zscore(df["Lead Time Reduction (%)"])
        + w_risk * zscore(df["Risk Reduction"])
        + w_profit * zscore(df["Profit Impact ($)"])
    )
    return df


def top_n_recommendations(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = scored[~scored["Is Home Factory"]].copy()
    candidates = candidates.sort_values(
        ["Product Name", "Region", "Composite Score"], ascending=[True, True, False]
    )
    ranked = candidates.groupby(["Product Name", "Region"]).head(top_n).copy()
    ranked["Rank"] = ranked.groupby(["Product Name", "Region"]).cumcount() + 1
    ranked["Recommend Reassignment"] = ranked["Composite Score"] > 0

    cols = [
        "Product Name",
        "Region",
        "Home Factory",
        "Candidate Factory",
        "Rank",
        "Recommend Reassignment",
        "Lead Time Reduction (%)",
        "Risk Reduction",
        "Profit Impact ($)",
        "Composite Score",
        "Current_Lead_Time_Days",
        "Predicted Lead Time Days",
        "Current_Margin_Pct",
        "Predicted Margin Pct",
    ]
    return ranked[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--w-leadtime", type=float, default=0.4)
    ap.add_argument("--w-risk", type=float, default=0.3)
    ap.add_argument("--w-profit", type=float, default=0.3)
    args = ap.parse_args()

    scenarios = pd.read_csv(os.path.join(args.outdir, "scenario_factory_reassignment.csv"))
    processed = pd.read_csv(os.path.join(args.outdir, "processed.csv"))

    risk = factory_risk_table(processed)
    scored = score_candidates(scenarios, risk, args.w_leadtime, args.w_risk, args.w_profit)
    recommendations = top_n_recommendations(scored, args.top_n)

    risk_path = os.path.join(args.outdir, "factory_risk_profile.csv")
    rec_path = os.path.join(args.outdir, "recommendations_factory_reassignment.csv")
    risk.to_csv(risk_path, index=False)
    recommendations.to_csv(rec_path, index=False)

    print(f"Wrote {risk_path}")
    print("=== Factory risk profile (lower CV = more consistent) ===")
    print(risk.to_string(index=False))

    print(f"\nWrote {rec_path} ({recommendations.shape})")
    print(f"\n=== Top-{args.top_n} factory reassignment candidates per Product x Region (recommended only) ===")
    print(
        recommendations[recommendations["Recommend Reassignment"]]
        .sort_values("Composite Score", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
