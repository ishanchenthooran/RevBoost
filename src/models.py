import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from .utils import cfg

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common telco columns -> generic names used below
    X = df.copy()
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    if "Churn" in X.columns:
        X["Churn"] = (X["Churn"].astype(str).str.lower().str.startswith("y")).astype(int)
    return X

def train_churn(df: pd.DataFrame):
    C = cfg()
    cols = C["model"]["features"]
    X = _prep(df)
    feats = [c for c in cols if c in X.columns]
    cat = [c for c in feats if X[c].dtype == "object"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    y = X["Churn"] if "Churn" in X.columns else (np.random.rand(len(X)) < 0.2).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X[feats], y, test_size=C["model"]["test_size"], random_state=7)
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=500))])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(X[feats])[:,1]
    return pipe, proba

def cluster_segments(df: pd.DataFrame):
    # pick numeric cols that likely exist
    cand = [c for c in ["tenure","MonthlyCharges","TotalCharges"] if c in df.columns]
    X = df[cand].copy()
    X["TotalCharges"] = pd.to_numeric(X.get("TotalCharges", 0), errors="coerce").fillna(0)
    km = KMeans(n_clusters=cfg()["model"]["n_clusters"], n_init="auto", random_state=7).fit(X)
    return km, km.labels_

def upsell_flags(df: pd.DataFrame) -> pd.Series:
    # simple heuristic: high usage/value and not already top tier
    plan_col = next((c for c in ["plan_tier","Contract"] if c in df.columns), None)
    is_mid = df[plan_col].isin(["Month-to-month","Basic","One year"]) if plan_col else True
    high_value = df.get("MonthlyCharges", pd.Series([0]*len(df))).ge(df["MonthlyCharges"].quantile(0.7) if "MonthlyCharges" in df else 50)
    return (is_mid & high_value).astype(int)
