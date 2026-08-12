"""
Step: Route & Product Clustering
-----------------------------------
Groups (Factory, Region, Product) combinations -- "routes" -- by performance
similarity (lead time, order volume, margin) using KMeans, then labels each
cluster in plain terms so downstream steps and reports can flag:
  - Consistently slow routes      (high lead time, regardless of volume)
  - Congested region-product combos (high volume AND slow lead time)
  - High-performing routes        (fast, healthy margin)
  - Low-volume / niche routes     (little data, low priority)

Cluster count is chosen once via silhouette score (not hardcoded), and labels
are derived from each cluster's centroid rather than assumed in advance, so
the labeling still makes sense if the underlying data changes.

Usage:
    python clustering.py --outdir out --k-min 2 --k-max 6
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Avg_Lead_Time_Days", "Orders", "Avg_Margin_Pct"]


def build_route_product_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["Factory", "Region", "Product Name"])
        .agg(
            Orders=("Row ID", "count"),
            Avg_Lead_Time_Days=("Lead Time Days", "mean"),
            Avg_Margin_Pct=("Margin Pct", "mean"),
            Total_Sales=("Sales", "sum"),
        )
        .reset_index()
    )
    return table


def choose_k(X_scaled: np.ndarray, k_min: int, k_max: int, random_state: int = 42):
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        if k >= len(X_scaled):
            break
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def label_clusters(table: pd.DataFrame) -> pd.DataFrame:
    centroids = table.groupby("Cluster")[FEATURES].mean()
    lead_time_rank = centroids["Avg_Lead_Time_Days"].rank(ascending=False)  # 1 = slowest
    volume_rank = centroids["Orders"].rank(ascending=False)  # 1 = highest volume
    margin_rank = centroids["Avg_Margin_Pct"].rank(ascending=False)  # 1 = highest margin

    n = len(centroids)
    labels = {}
    for cid in centroids.index:
        is_slow = lead_time_rank[cid] <= max(1, n // 2)
        is_congested = volume_rank[cid] <= max(1, n // 2)
        is_healthy_margin = margin_rank[cid] <= max(1, n // 2)

        if is_slow and is_congested:
            labels[cid] = "Congested High-Volume"
        elif is_slow:
            labels[cid] = "Consistently Slow"
        elif is_congested and is_healthy_margin:
            labels[cid] = "High-Performing"
        else:
            labels[cid] = "Low-Volume / Niche"

    table["Cluster Label"] = table["Cluster"].map(labels)
    return table, centroids.assign(**{"Cluster Label": pd.Series(labels)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.outdir, "processed.csv"))
    table = build_route_product_table(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(table[FEATURES])

    k, silhouette = choose_k(X_scaled, args.k_min, args.k_max)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    table["Cluster"] = kmeans.fit_predict(X_scaled)
    print(f"Chosen k={k} (silhouette={silhouette:.4f})")

    table, centroid_summary = label_clusters(table)

    route_products_path = os.path.join(args.outdir, "route_product_clusters.csv")
    table.sort_values(["Cluster Label", "Avg_Lead_Time_Days"], ascending=[True, False]).to_csv(
        route_products_path, index=False
    )
    print(f"Wrote {route_products_path} ({table.shape})")

    summary_path = os.path.join(args.outdir, "cluster_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "k": k,
                "silhouette_score": silhouette,
                "centroids": centroid_summary.reset_index().to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")

    print("\n=== Cluster sizes ===")
    print(table["Cluster Label"].value_counts().to_string())

    slow = table[table["Cluster Label"] == "Consistently Slow"]
    congested = table[table["Cluster Label"] == "Congested High-Volume"]
    print(f"\n=== Consistently slow routes ({len(slow)}) ===")
    print(slow[["Factory", "Region", "Product Name", "Orders", "Avg_Lead_Time_Days", "Avg_Margin_Pct"]].head(10).to_string(index=False))
    print(f"\n=== Congested region-product combos ({len(congested)}) ===")
    print(congested[["Factory", "Region", "Product Name", "Orders", "Avg_Lead_Time_Days", "Avg_Margin_Pct"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
