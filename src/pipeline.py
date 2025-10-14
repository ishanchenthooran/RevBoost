import os, logging, pandas as pd
from .utils import cfg, ensure_dirs
from .quality import dq_report, assert_minimums
from .models import train_churn, cluster_segments, upsell_flags

def run():
    C = cfg(); ensure_dirs()
    raw = C["paths"]["raw"]
    df = pd.read_csv(raw)

    rep = dq_report(df)
    logging.info(f"DATA QUALITY: {rep}")
    assert_minimums(rep)

    _, churn_p = train_churn(df)
    df["churn_prob"] = churn_p.round(4)

    _, seg = cluster_segments(df)
    df["segment"] = seg

    df["upsell_flag"] = upsell_flags(df)
    df["risk_score"] = (df["churn_prob"]*100).round(1)
    df["upsell_score"] = df["upsell_flag"]*100

    out_dir = C["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "revboost_scored.csv")
    df.to_csv(out, index=False)
    logging.info(f"Wrote {out} ({df.shape})")

if __name__ == "__main__":
    run()
