"""
Step 2: Predictive Modeling & Evaluation
------------------------------------------
Trains regressors on the encoded feature matrix from data_prep.py for two
targets:
  1. Lead Time Days   (shipping speed outcome)
  2. Margin Pct       (profitability outcome)

For each target, several candidate models spanning the accuracy /
interpretability spectrum are trained on the same train/test split and
scored with RMSE, MAE, and R2. The best-performing model is then selected
based on accuracy AND interpretability: among candidates whose R2 is within
BEST_R2_TOLERANCE of the top score, the most interpretable one wins (a tie
broken toward the simpler, more explainable model rather than automatically
taking the most complex one).

  RMSE - penalizes large errors more heavily (sensitive to outliers)
  MAE  - average absolute prediction error (easy to explain in the target's units)
  R2   - share of variance in the target explained by the model

These models let simulate.py ask "what would lead time / margin look like
under a different Ship Mode or Region assignment?" for a given order profile.

Usage:
    python modeling.py --outdir out
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

TARGETS = ["Lead Time Days", "Margin Pct"]

# Candidates ordered from most to least interpretable. interpretability_rank
# is a fixed modeling judgment (1 = easiest to explain to a stakeholder),
# not something derived from the fitted model.
CANDIDATE_MODELS = [
    {
        "name": "LinearRegression",
        "interpretability_rank": 1,
        "build": lambda: LinearRegression(),
    },
    {
        "name": "DecisionTree",
        "interpretability_rank": 2,
        "build": lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42),
    },
    {
        "name": "RandomForest",
        "interpretability_rank": 3,
        "build": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
    },
    {
        "name": "GradientBoosting",
        "interpretability_rank": 4,
        "build": lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    },
]

# how far below the best R2 a more interpretable model is still allowed to be
# and still win the selection
BEST_R2_TOLERANCE = 0.02


def evaluate_candidate(spec, X_train, X_test, y_train, y_test):
    model = spec["build"]()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": root_mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    return model, metrics


def select_best(results, tolerance=BEST_R2_TOLERANCE):
    """Pick the most interpretable model within `tolerance` R2 of the top score."""
    best_r2 = max(r["metrics"]["r2"] for r in results)
    in_range = [r for r in results if best_r2 - r["metrics"]["r2"] <= tolerance]
    return min(in_range, key=lambda r: r["interpretability_rank"])


def train_target(X, y, name: str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for spec in CANDIDATE_MODELS:
        model, metrics = evaluate_candidate(spec, X_train, X_test, y_train, y_test)
        print(
            f"[{name}] {spec['name']:<17} "
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
        )
        results.append({"name": spec["name"], "interpretability_rank": spec["interpretability_rank"],
                         "model": model, "metrics": metrics})

    best = select_best(results)
    print(f"[{name}] selected: {best['name']} (accuracy + interpretability tradeoff)\n")

    if hasattr(best["model"], "feature_importances_"):
        importances = (
            pd.Series(best["model"].feature_importances_, index=X.columns)
            .sort_values(ascending=False)
            .head(10)
        )
        print(f"[{name}] top features ({best['name']}):\n{importances.to_string()}\n")
    elif hasattr(best["model"], "coef_"):
        coefs = (
            pd.Series(best["model"].coef_, index=X.columns)
            .reindex(pd.Series(best["model"].coef_, index=X.columns).abs().sort_values(ascending=False).index)
            .head(10)
        )
        print(f"[{name}] top coefficients ({best['name']}):\n{coefs.to_string()}\n")

    comparison = [
        {"name": r["name"], "interpretability_rank": r["interpretability_rank"],
         "selected": r["name"] == best["name"], **r["metrics"]}
        for r in results
    ]
    return best["model"], best["name"], best["metrics"], comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    X = pd.read_csv(os.path.join(args.outdir, "features.csv"))
    df = pd.read_csv(os.path.join(args.outdir, "processed.csv"))

    metrics_all = {}
    comparison_all = {}
    for target in TARGETS:
        y = df[target]
        model, model_name, metrics, comparison = train_target(X, y, target)
        metrics_all[target] = {"selected_model": model_name, **metrics}
        comparison_all[target] = comparison

        model_path = os.path.join(args.outdir, f"model_{target.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {model_path} ({model_name})")

    with open(os.path.join(args.outdir, "model_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    with open(os.path.join(args.outdir, "model_comparison.json"), "w") as f:
        json.dump(comparison_all, f, indent=2)
    print(f"Wrote {os.path.join(args.outdir, 'model_comparison.json')}")


if __name__ == "__main__":
    main()
