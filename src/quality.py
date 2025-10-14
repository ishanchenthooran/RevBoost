import pandas as pd
from .utils import cfg

def dq_report(df: pd.DataFrame) -> dict:
    rep = {
        "rows": len(df),
        "nulls": df.isna().sum().to_dict()
    }
    # basic guards if you keep telco schema
    if "MonthlyCharges" in df.columns:
        rep["monthly_nonneg"] = int((df["MonthlyCharges"] >= 0).all())
    if "TotalCharges" in df.columns:
        rep["total_nonneg"] = int(pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).ge(0).all())
    return rep

def assert_minimums(rep: dict):
    if rep["rows"] < cfg()["quality"]["min_rows"]:
        raise ValueError(f"Too few rows: {rep['rows']}")
