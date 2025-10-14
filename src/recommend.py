import pandas as pd

def rec_for_row(r: pd.Series) -> str:
    tips = []
    if r.get("churn_prob", 0) > 0.6: tips.append("Send save-offer + schedule success call")
    if r.get("MonthlyCharges", 0) < 25: tips.append("Promote usage quick wins")
    if r.get("upsell_flag", 0) == 1 and r.get("churn_prob", 0) < 0.3: tips.append("Suggest plan upgrade")
    return " • ".join(tips) if tips else "Review account notes"
