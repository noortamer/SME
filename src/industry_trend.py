# industry_trend.py
import pandas as pd
import numpy as np

INPUT  = r"data\finalscores.csv"
OUTPUT = r"data\industry_trends.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig")
df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
df["score"] = df["score"].astype(float)
df["السنة"] = pd.to_numeric(df["السنة"], errors="coerce")

# Industry-year average score (unweighted mean of companies)
ind = (
    df.groupby(["القطاع", "السنة"], dropna=False)["score"]
      .mean()
      .reset_index(name="industry_score")
      .sort_values(["القطاع", "السنة"])
)

# YoY change
ind["industry_score_prev"] = ind.groupby("القطاع")["industry_score"].shift(1)
ind["industry_yoy_change"] = (ind["industry_score"] - ind["industry_score_prev"]).round(4)

# 3-year rolling average & slope (trend)
ind["industry_score_3yr_avg"] = (
    ind.groupby("القطاع")["industry_score"].rolling(3).mean().reset_index(0, drop=True).round(4)
)

def three_year_slope(s):
    if len(s) < 3: return np.nan
    x = np.arange(len(s[-3:]))
    y = s[-3:]
    x_mean, y_mean = x.mean(), y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return (num / den) if den != 0 else np.nan

ind["industry_score_3yr_slope"] = (
    ind.groupby("القطاع")["industry_score"]
      .apply(lambda s: s.rolling(3).apply(lambda w: three_year_slope(pd.Series(w)), raw=False))
      .reset_index(level=0, drop=True)
      .round(4)
)

# (Optional) Rank industries within each year by their industry_score to see annual position
ind["industry_rank_within_year"] = (
    ind.groupby("السنة")["industry_score"].rank(method="average", ascending=False).astype(int)
)

ind.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved industry trends to {OUTPUT}")