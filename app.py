"""
Streamlit dashboard for the Factory Reallocation & Shipping Optimization
pipeline (data_prep -> modeling -> clustering -> simulate* -> recommend* ->
kpis). Reads the CSV/JSON artifacts each script already writes to --outdir;
it does not re-implement any of the modeling or scoring logic.

Usage:
    streamlit run app.py
"""
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from viz_utils import (
    BLUE, CLUSTER_COLORS, FACTORY_COLORS, MODEL_COLORS, SHIP_MODE_COLORS,
    categorical_bar, categorical_scatter, delta_bar_chart, kpi_tile,
    load_csv, load_json, outputs_exist, status_color, style_fig,
)
import plotly.graph_objects as go

st.set_page_config(page_title="Factory Reallocation & Shipping Optimization", layout="wide")

# --------------------------------------------------------------------------
# Sidebar: pipeline controls
# --------------------------------------------------------------------------
st.sidebar.title("Pipeline")
outdir = st.sidebar.text_input("Output directory", value="out")
input_csv = st.sidebar.text_input("Input CSV", value="Nassau Candy Distributor.csv")
top_n = st.sidebar.number_input("Top-N factory candidates", min_value=1, max_value=5, value=2)

if st.sidebar.button("Run full pipeline", type="primary"):
    with st.sidebar.status("Running pipeline...", expanded=True) as status:
        result = subprocess.run(
            [sys.executable, "pipeline.py", "--input", input_csv, "--outdir", outdir, "--top-n", str(top_n)],
            capture_output=True, text=True,
        )
        st.sidebar.code((result.stdout or "") + (result.stderr or ""), language="text")
        if result.returncode == 0:
            status.update(label="Pipeline finished", state="complete")
            st.cache_data.clear()
        else:
            status.update(label="Pipeline failed", state="error")
    st.rerun()

st.sidebar.caption("Runs data_prep -> modeling -> clustering -> simulate* -> recommend* -> kpis and writes every artifact into the output directory above.")

st.title("Factory Reallocation & Shipping Optimization")

if not outputs_exist(outdir):
    st.warning(f"No pipeline outputs found in `{outdir}`. Click **Run full pipeline** in the sidebar to generate them.")
    st.stop()

# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
baseline = load_csv(outdir, "baseline_by_factory_region.csv")
model_comparison = load_json(outdir, "model_comparison.json") or {}
model_metrics = load_json(outdir, "model_metrics.json") or {}
clusters = load_csv(outdir, "route_product_clusters.csv")
cluster_summary = load_json(outdir, "cluster_summary.json") or {}
scenario_ship = load_csv(outdir, "scenario_results.csv")
scenario_factory = load_csv(outdir, "scenario_factory_reassignment.csv")
rec_ship = load_csv(outdir, "recommendations_ship_mode.csv")
rec_realloc = load_csv(outdir, "recommendations_reallocation.csv")
rec_factory = load_csv(outdir, "recommendations_factory_reassignment.csv")
factory_risk = load_csv(outdir, "factory_risk_profile.csv")
kpi_detail = load_csv(outdir, "kpi_detail.csv")
kpi_summary = load_json(outdir, "kpi_summary.json") or {}

tab_overview, tab_models, tab_clusters, tab_scenarios, tab_recs, tab_kpis = st.tabs(
    ["Overview", "Model Evaluation", "Route & Product Clustering", "Scenario Simulation", "Recommendations", "KPIs"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    st.markdown(
        "Predicts Lead Time and Margin per order, clusters routes/products by "
        "performance, simulates ship-mode and factory-reassignment scenarios, "
        "then ranks and recommends the changes with the best lead-time, risk, "
        "and profit tradeoff."
    )

    if kpi_summary:
        cols = st.columns(4)
        with cols[0]:
            v = kpi_summary.get("avg_lead_time_reduction_pct")
            kpi_tile("Avg Lead Time Reduction", f"{v:.1f}%" if v is not None else "n/a",
                      "across recommended reassignments", status_color(v or 0, good=10, warning=0))
        with cols[1]:
            v = kpi_summary.get("avg_profit_impact_stability")
            kpi_tile("Profit Impact Stability", f"{v:.2f}" if v is not None else "n/a",
                      "1.0 = fully stable historical margin", status_color(v or 0, good=0.7, warning=0.4))
        with cols[2]:
            v = kpi_summary.get("avg_scenario_confidence_score")
            kpi_tile("Scenario Confidence", f"{v:.2f}" if v is not None else "n/a",
                      "model fit x data volume behind the route", status_color(v or 0, good=0.5, warning=0.25))
        with cols[3]:
            v = kpi_summary.get("total_estimated_profit_impact")
            kpi_tile("Total Est. Profit Impact", f"${v:,.0f}" if v is not None else "n/a",
                      f"{kpi_summary.get('recommended_reassignments', 0)} recommended moves", BLUE)

        r2_lt = kpi_summary.get("model_quality_leadtime_r2")
        if r2_lt is not None and r2_lt < 0.3:
            st.warning(
                f"The Lead Time model's R² is only {r2_lt:.3f} -- lead time is barely explained by the "
                "available features, so Scenario Confidence and lead-time-driven recommendations should "
                "be read as directional, not precise."
            )

    st.subheader("Current baseline by Factory x Region")
    st.dataframe(baseline, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Model Evaluation
# --------------------------------------------------------------------------
with tab_models:
    st.caption("Best model per target is selected on accuracy (R²) with a tie-break toward interpretability.")
    for target, candidates in model_comparison.items():
        st.subheader(target)
        df = pd.DataFrame(candidates)
        selected = model_metrics.get(target, {})

        m1, m2, m3 = st.columns(3)
        m1.metric("Selected model", selected.get("selected_model", "n/a"))
        m2.metric("R²", f"{selected.get('r2', 0):.4f}")
        m3.metric("MAE", f"{selected.get('mae', 0):.4f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = categorical_bar(df, "name", "r2", "name", MODEL_COLORS, "R² by candidate model", "R²")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = categorical_bar(df, "name", "mae", "name", MODEL_COLORS, "MAE by candidate model", "MAE")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        display_df = df.copy()
        display_df["selected"] = display_df["selected"].map({True: "✓ selected", False: ""})
        st.dataframe(
            display_df[["name", "interpretability_rank", "mae", "rmse", "r2", "selected"]],
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------
# Route & Product Clustering
# --------------------------------------------------------------------------
with tab_clusters:
    if clusters.empty:
        st.info("No cluster output found. Run the pipeline first.")
    else:
        k = cluster_summary.get("k")
        sil = cluster_summary.get("silhouette_score")
        st.caption(f"k={k} clusters chosen by silhouette score ({sil:.3f})" if k else "")

        f1, f2, f3 = st.columns(3)
        factories = f1.multiselect("Factory", sorted(clusters["Factory"].unique()), default=list(clusters["Factory"].unique()))
        regions = f2.multiselect("Region", sorted(clusters["Region"].unique()), default=list(clusters["Region"].unique()))
        labels = f3.multiselect("Cluster", sorted(clusters["Cluster Label"].unique()), default=list(clusters["Cluster Label"].unique()))

        filtered = clusters[
            clusters["Factory"].isin(factories) & clusters["Region"].isin(regions) & clusters["Cluster Label"].isin(labels)
        ]

        c1, c2 = st.columns([1, 2])
        with c1:
            counts = filtered["Cluster Label"].value_counts().reset_index()
            counts.columns = ["Cluster Label", "Count"]
            fig = categorical_bar(counts, "Cluster Label", "Count", "Cluster Label", CLUSTER_COLORS, "Routes per cluster", "Routes")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = categorical_scatter(
                filtered, "Orders", "Avg_Lead_Time_Days", "Cluster Label", CLUSTER_COLORS,
                "Order volume vs. lead time by cluster",
                hover_data=["Factory", "Region", "Product Name", "Avg_Margin_Pct"],
                size_col="Total_Sales",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Flagged routes")
        st.dataframe(
            filtered.sort_values(["Cluster Label", "Avg_Lead_Time_Days"], ascending=[True, False]),
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------
# Scenario Simulation
# --------------------------------------------------------------------------
with tab_scenarios:
    mode = st.radio("Scenario type", ["Ship Mode (per Factory x Region)", "Factory Reassignment (per Product x Region)"], horizontal=True)

    if mode.startswith("Ship Mode"):
        if scenario_ship.empty:
            st.info("No ship-mode scenario output found. Run the pipeline first.")
        else:
            c1, c2 = st.columns(2)
            factory = c1.selectbox("Factory", sorted(scenario_ship["Factory"].unique()))
            regions_avail = sorted(scenario_ship.loc[scenario_ship["Factory"] == factory, "Region"].unique())
            region = c2.selectbox("Region", regions_avail)

            subset = scenario_ship[(scenario_ship["Factory"] == factory) & (scenario_ship["Region"] == region)].copy()
            subset["Ship Mode Label"] = subset.apply(
                lambda r: f"{r['Ship Mode']} (current)" if r["Is Current Ship Mode"] else r["Ship Mode"], axis=1
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = categorical_bar(subset, "Ship Mode Label", "Predicted Lead Time Days", "Ship Mode", SHIP_MODE_COLORS, "Predicted lead time by ship mode", "Days")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = categorical_bar(subset, "Ship Mode Label", "Predicted Margin Pct", "Ship Mode", SHIP_MODE_COLORS, "Predicted margin by ship mode", "Margin")
                st.plotly_chart(fig, use_container_width=True)

            fig = delta_bar_chart(subset, "Ship Mode Label", "Delta Lead Time Days", "Lead time change vs. current ship mode", "Δ Days")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(subset.drop(columns=["Ship Mode Label"]), use_container_width=True, hide_index=True)
    else:
        if scenario_factory.empty:
            st.info("No factory-reassignment scenario output found. Run the pipeline first.")
        else:
            c1, c2 = st.columns(2)
            product = c1.selectbox("Product", sorted(scenario_factory["Product Name"].unique()))
            regions_avail = sorted(scenario_factory.loc[scenario_factory["Product Name"] == product, "Region"].unique())
            region = c2.selectbox("Region", regions_avail)

            subset = scenario_factory[(scenario_factory["Product Name"] == product) & (scenario_factory["Region"] == region)].copy()
            subset["Factory Label"] = subset.apply(
                lambda r: f"{r['Candidate Factory']} (home)" if r["Is Home Factory"] else r["Candidate Factory"], axis=1
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = categorical_bar(subset, "Factory Label", "Predicted Lead Time Days", "Candidate Factory", FACTORY_COLORS, "Predicted lead time by candidate factory", "Days")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = categorical_bar(subset, "Factory Label", "Predicted Margin Pct", "Candidate Factory", FACTORY_COLORS, "Predicted margin by candidate factory", "Margin")
                st.plotly_chart(fig, use_container_width=True)

            fig = delta_bar_chart(subset, "Factory Label", "Delta Lead Time Days", "Lead time change vs. current (home) factory", "Δ Days")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(subset.drop(columns=["Factory Label"]), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
with tab_recs:
    st.subheader("Ship Mode recommendations")
    if rec_ship.empty:
        st.info("No ship-mode recommendations found.")
    else:
        only_switches = st.checkbox("Show only recommended switches", value=True, key="ship_switch_filter")
        df = rec_ship[rec_ship["Recommend Switch"]] if only_switches else rec_ship
        st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Regional reallocation flags")
    if rec_realloc.empty:
        st.info("No reallocation flags found.")
    else:
        only_flagged = st.checkbox("Show only flagged regions", value=True, key="realloc_filter")
        df = rec_realloc[rec_realloc["Reallocation Candidate"]] if only_flagged else rec_realloc
        st.dataframe(df.sort_values(["Factory", "Performance Score"]), use_container_width=True, hide_index=True)

    st.subheader("Factory reassignment recommendations")
    if rec_factory.empty:
        st.info("No factory reassignment recommendations found.")
    else:
        only_recommended = st.checkbox("Show only recommended reassignments", value=True, key="factory_rec_filter")
        df = rec_factory[rec_factory["Recommend Reassignment"]] if only_recommended else rec_factory
        df = df.sort_values("Composite Score", ascending=False)

        top = df.head(10).copy()
        top["Label"] = top["Product Name"] + " (" + top["Region"] + "): " + top["Home Factory"] + " -> " + top["Candidate Factory"]
        fig = go.Figure(go.Bar(x=top["Label"], y=top["Lead Time Reduction (%)"], marker_color=BLUE))
        fig.update_layout(title="Top-10 reassignments by composite score", yaxis_title="Lead Time Reduction (%)", xaxis_tickangle=-30)
        st.plotly_chart(style_fig(fig), use_container_width=True)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("**Factory risk profile** (lower CV = more historically consistent)")
        st.dataframe(factory_risk, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
with tab_kpis:
    if not kpi_summary:
        st.info("No KPI summary found. Run the pipeline first.")
    else:
        cols = st.columns(3)
        with cols[0]:
            v = kpi_summary.get("avg_lead_time_reduction_pct")
            kpi_tile("Lead Time Reduction (%)", f"{v:.1f}%" if v is not None else "n/a", "operational gain", status_color(v or 0, good=10, warning=0))
        with cols[1]:
            v = kpi_summary.get("avg_profit_impact_stability")
            kpi_tile("Profit Impact Stability", f"{v:.2f}" if v is not None else "n/a", "financial safety", status_color(v or 0, good=0.7, warning=0.4))
        with cols[2]:
            v = kpi_summary.get("avg_scenario_confidence_score")
            kpi_tile("Scenario Confidence Score", f"{v:.2f}" if v is not None else "n/a", "trust in the simulated scenario", status_color(v or 0, good=0.5, warning=0.25))

        st.markdown("")
        st.json(kpi_summary)

        if not kpi_detail.empty:
            st.subheader("Per-recommendation detail")
            sort_col = st.selectbox(
                "Sort by",
                ["Scenario Confidence Score", "Lead Time Reduction (%)", "Profit Impact Stability", "Profit Impact ($)", "Composite Score"],
                index=0,
            )
            st.dataframe(kpi_detail.sort_values(sort_col, ascending=False), use_container_width=True, hide_index=True)
