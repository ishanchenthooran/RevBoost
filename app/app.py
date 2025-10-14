import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.summarizer import generate_summary
from src.clustering import perform_kmeans_clustering
from src.predict_churn import train_logistic_model

PROCESSED_PATH = "data/processed/revboost_scored.csv"
RAW_PATH = "data/cleaned_telco.csv"

# ------------------------------- Helpers -------------------------------

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize names from pipeline vs in-app compute and create string segment for UI."""
    if "churn_prob" in df.columns and "Churn_Prob" not in df.columns:
        df["Churn_Prob"] = df["churn_prob"]
    if "segment" in df.columns and "Segment" not in df.columns:
        df["Segment"] = df["segment"]
    if "customerID" not in df.columns:
        df["customerID"] = [f"CUST{str(i).zfill(4)}" for i in range(len(df))]
    # Coerce TotalCharges to numeric if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # Keep numeric Segment for computations, add string copy for UI/plots
    if "Segment" in df.columns:
        df["_Segment_str"] = df["Segment"].astype(str)
    else:
        df["Segment"] = 0
        df["_Segment_str"] = "0"
    return df

def add_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    df["Churn_Prob"] = pd.to_numeric(df["Churn_Prob"], errors="coerce")
    df = df[df["Churn_Prob"].between(0, 1)].copy()

    def risk_level(p):
        if p > 0.7: return "High"
        if p > 0.4: return "Medium"
        return "Low"

    df["Risk_Level"] = df["Churn_Prob"].apply(risk_level)
    return df

def infer_upsell_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential upsell candidates based on flexible contracts,
    moderate-to-high spending, and moderate churn stability.
    """
    if "upsell_flag" in df.columns:
        return df

    # Safely coerce numerics
    for col in ["MonthlyCharges", "TotalServicesUsed", "Churn_Prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Thresholds — slightly tighter
    monthly_thresh = df["MonthlyCharges"].quantile(0.65)  # top 35% spenders
    churn_thresh = 0.5  # low-to-medium churn risk
    max_services = df["TotalServicesUsed"].max() if "TotalServicesUsed" in df.columns else 0

    # Indicators
    is_monthly = (
        df["Contract"].astype(str).str.lower().str.contains("month")
        if "Contract" in df.columns else pd.Series(False, index=df.index)
    )
    spends_enough = df["MonthlyCharges"] >= monthly_thresh
    moderate_churn = df["Churn_Prob"] <= churn_thresh
    has_room_to_add = df["TotalServicesUsed"] < max_services

    # Weighted scoring system
    score = (
        (is_monthly.astype(int) * 1.0) +
        (spends_enough.astype(int) * 1.5) +
        (moderate_churn.astype(int) * 1.0) +
        (has_room_to_add.astype(int) * 1.0)
    )

    # Flag as upsell if total score ≥ 3 (so only best-fit customers)
    df["upsell_flag"] = (score >= 3).astype(int)

    return df

# ------------------------------- Load data -------------------------------

def load_data() -> tuple[pd.DataFrame, str]:
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
        source = "processed"
    else:
        # Fallback: your current workflow
        df = pd.read_csv(RAW_PATH)

        if "customerID" not in df.columns:
            df["customerID"] = [f"CUST{str(i).zfill(4)}" for i in range(len(df))]

        clustering_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "TotalServicesUsed"]
        use_features = [c for c in clustering_features if c in df.columns]
        if use_features:
            df, _ = perform_kmeans_clustering(df, use_features, k=3)
        else:
            df["Segment"] = 0  # safe default

        churn_features = ["tenure", "MonthlyCharges", "TotalServicesUsed", "Contract"]
        use_churn = [c for c in churn_features if c in df.columns]
        if use_churn:
            df, _ = train_logistic_model(df, use_churn, target="Churn")
        else:
            df["Churn_Prob"] = 0.1

        source = "raw+in-app"

    return normalize_schema(df), source

# ------------------------------- UI -------------------------------

st.set_page_config(page_title="RevBoost Dashboard", layout="wide")
st.title("📊 RevBoost: Customer Segmentation & Churn Insights")

df, source = load_data()

# Clean + enrich
df = add_risk_levels(df)
df = infer_upsell_flag(df)

# ------------------------------- KPIs -------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", len(df))
k2.metric("Avg Churn %", round(df["Churn_Prob"].mean() * 100, 1) if len(df) else 0)
k3.metric("High Risk (≥70%)", int((df["Churn_Prob"] >= 0.7).sum()))
k4.metric("Upsell Leads", int(df["upsell_flag"].sum()))

# ------------------------------- Cluster Insights -------------------------------
st.header("📌 Cluster Insights")

plot_df = df.copy()
plot_df["Segment"] = plot_df["_Segment_str"]  # use string for plotting

avg_churn = plot_df.groupby("_Segment_str", as_index=False)["Churn_Prob"].mean()
fig = px.bar(
    avg_churn,
    x="_Segment_str",
    y="Churn_Prob",
    color="_Segment_str",
    text=avg_churn["Churn_Prob"].apply(lambda x: f"{x:.2f}"),
    title="Average Churn Probability by Segment",
    template="plotly_dark"
)
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Top Risk Table -------------------------------
st.header("⚠️ Top At-Risk Customers")

# Ensure the helper string column exists
if "_Segment_str" not in df.columns:
    if "Segment" in df.columns:
        df["_Segment_str"] = df["Segment"].astype(str)
    else:
        df["_Segment_str"] = "0"

# Build risk table
risk_cols = [
    c for c in [
        "customerID", "_Segment_str", "Churn_Prob", "Risk_Level",
        "Contract", "TotalServicesUsed", "MonthlyCharges"
    ] if c in df.columns
]

top_risk = df.sort_values("Churn_Prob", ascending=False).copy()

display_risk = (
    top_risk[risk_cols]
    .rename(columns={"_Segment_str": "Segment"})
    .reset_index(drop=True)
)

st.dataframe(display_risk, height=320, use_container_width=True)

# ------------------------------- Upsell Table -------------------------------
st.header("⬆️ Top Upsell Opportunities")
ups = df[df["upsell_flag"] == 1].copy()
ups = ups.sort_values(["upsell_flag", "MonthlyCharges" if "MonthlyCharges" in df.columns else "Churn_Prob"],
                      ascending=[False, False]).head(200)
ups_cols = [c for c in ["customerID", "_Segment_str", "MonthlyCharges", "Contract", "Churn_Prob"] if c in df.columns]
if len(ups):
    st.dataframe(ups[ups_cols].rename(columns={"_Segment_str": "Segment"}).reset_index(drop=True),
                 height=320, use_container_width=True)
else:
    st.info("No upsell candidates found under current heuristic.")

# ------------------------------- Strategy (LLM-based) -------------------------------
st.header("🧐 LLM-Based Retention Strategy")
segment_descriptions = {
    0: "Segment 0 – Price sensitive customers with short tenure.",
    1: "Segment 1 – Loyal customers with multiple services.",
    2: "Segment 2 – Newer users with high monthly charges."
}

strategy_mode = st.radio("Choose strategy view:", ["Individual Customer", "Customer Segment"])

if strategy_mode == "Individual Customer":
    selected_id = st.selectbox("Select a Customer:", df["customerID"].unique())
    row = df[df["customerID"] == selected_id].iloc[0]

    # safe numeric cast for Segment
    seg_val = row.get("Segment", 0)
    try:
        seg_int = int(seg_val)
    except Exception:
        seg_int = 0

    profile = {
        "Segment": seg_int,
        "tenure": int(row["tenure"]) if "tenure" in df.columns else None,
        "MonthlyCharges": float(row["MonthlyCharges"]) if "MonthlyCharges" in df.columns else None,
        "TotalServicesUsed": int(row["TotalServicesUsed"]) if "TotalServicesUsed" in df.columns else None,
        "Churn_Prob": round(float(row["Churn_Prob"]), 3)
    }

    with st.expander("📄 View Customer Profile"):
        st.json(profile)

    if st.button("🔍 Generate Customer Strategy"):
        with st.spinner("Analyzing..."):
            recommendation = generate_summary(profile)
            st.success("Strategy Generated!")
            st.markdown("### 📋 Recommendation:")
            st.write(recommendation)

else:
    # UI uses string values, map back to numeric safely
    seg_choices = sorted(df["_Segment_str"].unique())
    segment_id_str = st.selectbox("Select a Segment:", seg_choices)

    seg_df = df[df["_Segment_str"] == segment_id_str]
    seg_value_raw = seg_df["Segment"].iloc[0] if len(seg_df) else 0
    try:
        seg_value_int = int(seg_value_raw)
    except Exception:
        seg_value_int = 0

    avg_profile = {
        "Segment": seg_value_int,
        "tenure": int(seg_df["tenure"].mean()) if "tenure" in seg_df.columns and len(seg_df) else None,
        "MonthlyCharges": float(seg_df["MonthlyCharges"].mean()) if "MonthlyCharges" in seg_df.columns and len(seg_df) else None,
        "TotalServicesUsed": int(seg_df["TotalServicesUsed"].mean()) if "TotalServicesUsed" in seg_df.columns and len(seg_df) else None,
        "Churn_Prob": round(float(seg_df["Churn_Prob"].mean()), 3) if "Churn_Prob" in seg_df.columns and len(seg_df) else None,
    }

    st.markdown(f"**📎 Segment Description:** {segment_descriptions.get(seg_value_int, 'No description available.')}")

    with st.expander("📄 View Segment Profile"):
        st.json(avg_profile)

    if st.button("🧠 Generate Segment Strategy"):
        with st.spinner("Analyzing segment profile..."):
            recommendation = generate_summary(avg_profile)
            st.success("Strategy Generated!")
            st.markdown("### 📋 Segment-Level Recommendation:")
            st.write(recommendation)

# ------------------------------- Export -------------------------------
st.download_button(
    "Download current view (CSV)",
    data=df.to_csv(index=False),
    file_name="revboost_view.csv",
    mime="text/csv"
)