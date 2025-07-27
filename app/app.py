import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.summarizer import generate_summary
from src.clustering import perform_kmeans_clustering
from src.predict_churn import train_logistic_model

# ---------- Load & Prepare Data ----------
df = pd.read_csv('data/cleaned_telco.csv')

# Add synthetic customerID (if dropped during cleaning)
df['customerID'] = [f"CUST{str(i).zfill(4)}" for i in range(len(df))]

# Clustering
clustering_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServicesUsed']
df, _ = perform_kmeans_clustering(df, clustering_features, k=3)

# Churn Model
churn_features = ['tenure', 'MonthlyCharges', 'TotalServicesUsed', 'Contract']
df, model = train_logistic_model(df, churn_features)

# Ensure Churn_Prob is numeric and drop NaNs
df['Churn_Prob'] = pd.to_numeric(df['Churn_Prob'], errors='coerce')
df = df.dropna(subset=['Churn_Prob'])

# Add churn risk level
def risk_level(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

df['Risk_Level'] = df['Churn_Prob'].apply(risk_level)

# Segment Descriptions
segment_descriptions = {
    0: "Segment 0 – Price sensitive customers with short tenure.",
    1: "Segment 1 – Loyal customers with multiple services.",
    2: "Segment 2 – Newer users with high monthly charges."
}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RevBoost Dashboard", layout="wide")
st.title("📊 RevBoost: Customer Segmentation & Churn Insights")

# -------- Cluster Insights --------
st.header("📌 Cluster Insights")
# Remove invalid Churn_Prob values (NaN, inf, out of range)
df = df[df['Churn_Prob'].between(0, 1)]
fig = px.box(
    df, 
    x='Segment', 
    y='Churn_Prob', 
    color='Segment', 
    title='Churn Probability by Segment'
)
st.plotly_chart(fig, use_container_width=True)

# -------- Top Risk Table --------
st.header("⚠️ Top At-Risk Customers")
top_risk = df.sort_values('Churn_Prob', ascending=False)
st.dataframe(top_risk[['customerID', 'Segment', 'Churn_Prob', 'Risk_Level', 'Contract', 'TotalServicesUsed']].reset_index(drop=True), height=300)

# -------- Strategy Toggle --------
st.header("🧐 LLM-Based Retention Strategy")
strategy_mode = st.radio("Choose strategy view:", ["Individual Customer", "Customer Segment"])

if strategy_mode == "Individual Customer":
    selected_id = st.selectbox("Select a Customer:", df['customerID'].unique())
    customer_data = df[df['customerID'] == selected_id].iloc[0]
    profile = {
        "Segment": int(customer_data["Segment"]),
        "tenure": int(customer_data["tenure"]),
        "MonthlyCharges": float(customer_data["MonthlyCharges"]),
        "TotalServicesUsed": int(customer_data["TotalServicesUsed"]),
        "Churn_Prob": round(float(customer_data["Churn_Prob"]), 2)
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
    segment_id = st.selectbox("Select a Segment:", sorted(df['Segment'].unique()))
    segment_df = df[df['Segment'] == segment_id]

    if segment_id is not None:
        avg_profile = {
            "Segment": int(segment_id),
            "tenure": int(segment_df['tenure'].mean()),
            "MonthlyCharges": float(segment_df['MonthlyCharges'].mean()),
            "TotalServicesUsed": int(segment_df['TotalServicesUsed'].mean()),
            "Churn_Prob": round(float(segment_df['Churn_Prob'].mean()), 2)
        }

        st.markdown(f"**📎 Segment Description:** {segment_descriptions.get(segment_id, 'No description available.')}")

        with st.expander("📄 View Segment Profile"):
            st.json(avg_profile)

        if st.button("🧠 Generate Segment Strategy"):
            with st.spinner("Analyzing segment profile..."):
                recommendation = generate_summary(avg_profile)
                st.success("Strategy Generated!")
                st.markdown("### 📋 Segment-Level Recommendation:")
                st.write(recommendation)
