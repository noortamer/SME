# industry_benchmark.py
import pandas as pd

INPUT = r"data\scores_peers.csv"
OUTPUT = r"data\industry_benchmarks.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig")
df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
df["score"] = df["score"].astype(float)

# Industry-year average score
ind = (
    df.groupby(["القطاع", "السنة"], dropna=False)["score"]
      .mean()
      .reset_index(name="industry_score")
)

# Rank industries within each year (higher is better)
ind["industry_rank_within_year"] = (
    ind.groupby("السنة")["industry_score"].rank(method="average", ascending=False).astype(int)
)

# Percentile within year
ind["industry_percentile_within_year"] = (
    ind.groupby("السنة")["industry_score"].rank(pct=True) * 100
).round(2)

# (Optional) also compute the year’s cross-industry mean/median for context
yr = (
    ind.groupby("السنة")["industry_score"]
      .agg(year_industry_mean="mean", year_industry_median="median")
      .reset_index()
)
ind = ind.merge(yr, on="السنة", how="left")

ind = ind.sort_values(["السنة", "industry_rank_within_year"])
ind.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved industry benchmarks to {OUTPUT}")