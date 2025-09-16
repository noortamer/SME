# company_benchmark.py
import pandas as pd
import numpy as np

INPUT = r"data\scores_peers.csv"
OUTPUT = r"data\company_benchmarks.csv"

# Read scores (expects columns: 'القطاع' (sector), 'السنة' (year), 'score')
df = pd.read_csv(INPUT, encoding="utf-8-sig")

# Filter valid scores
df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
df["score"] = df["score"].astype(float)

# Group by industry-year
g = df.groupby(["القطاع", "السنة"], dropna=False)

# Industry-year stats
df["industry_mean"]   = g["score"].transform("mean").round(4)
df["industry_median"] = g["score"].transform("median").round(4)
df["industry_std"]    = g["score"].transform("std").round(4)

# Gaps and standardize
df["gap_to_mean"] = (df["score"] - df["industry_mean"]).round(4)
df["z_within_industry_year"] = (
    (df["score"] - df["industry_mean"]) / df["industry_std"].replace(0, np.nan)
).round(4)

# Rank within industry-year (higher score = better)
df["rank_within_industry_year"] = g["score"].rank(method="average", ascending=False).astype(int)
df["count_in_industry_year"] = g["score"].transform("count").astype(int)
df["percentile_within_industry_year"] = (
    g["score"].rank(pct=True, ascending=True) * 100
).round(2)

# Optional: keep only useful columns
keep = [
    "السنة", "القطاع", "score",
    "industry_mean", "industry_median", "industry_std",
    "gap_to_mean", "z_within_industry_year",
    "rank_within_industry_year", "count_in_industry_year",
    "percentile_within_industry_year", "الرقم_الضريبي"
]
# Include IDs/names if present
for c in ["company_id", "company_name", "اسم_الشركة", "معرف_الشركة", "فئة_SME"]:
    if c in df.columns: keep.insert(0, c)

out = df[keep].sort_values(["السنة", "القطاع", "rank_within_industry_year"])
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved company benchmarks to {OUTPUT}")