"""
Step 2: Predictive Modeling
----------------------------
Trains two regressors on the encoded feature matrix from data_prep.py:
  1. Lead Time Days   (shipping speed outcome)
  2. Margin Pct       (profitability outcome)

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

TARGETS = ["Lead Time Days", "Margin Pct"]


def train_target(X, y, name: str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": root_mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    print(f"[{name}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}")

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"[{name}] top features:\n{importances.to_string()}\n")

    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    X = pd.read_csv(os.path.join(args.outdir, "features.csv"))
    df = pd.read_csv(os.path.join(args.outdir, "processed.csv"))

    metrics_all = {}
    for target in TARGETS:
        y = df[target]
        model, metrics = train_target(X, y, target)
        metrics_all[target] = metrics

        model_path = os.path.join(args.outdir, f"model_{target.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {model_path}")

    with open(os.path.join(args.outdir, "model_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)


if __name__ == "__main__":
    main()
