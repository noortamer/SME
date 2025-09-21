import pandas as pd
import numpy as np

INPUT  = r"data\finalscores.csv"
OUTPUT = r"data\company_vs_size.csv"

df = pd.read_csv(INPUT, encoding="utf-8-sig")
df = df[pd.to_numeric(df["score"], errors="coerce").notna()].copy()
df["score"] = df["score"].astype(float)

# If size class missing, bail with a friendly message
if "فئة_SME" not in df.columns:
    raise ValueError("Column 'فئة_SME' (size class) is required for size benchmarking.")

g = df.groupby(["القطاع", "السنة", "فئة_SME"], dropna=False)

# Size-segment stats
df["size_mean"]   = g["score"].transform("mean").round(4)
df["size_median"] = g["score"].transform("median").round(4)
df["size_std"]    = g["score"].transform("std").round(4)
df["gap_to_size_mean"] = (df["score"] - df["size_mean"]).round(4)

# Rank & percentile within size segment
df["rank_within_size_segment"] = g["score"].rank(method="average", ascending=False).astype(int)
df["count_in_size_segment"]    = g["score"].transform("count").astype(int)
df["percentile_within_size_segment"] = (g["score"].rank(pct=True) * 100).round(2)

keep = []
for c in ["company_id", "company_name", "اسم_الشركة", "معرف_الشركة"]:
    if c in df.columns: keep.append(c)

keep += ["السنة", "القطاع", "فئة_SME", "score",
         "size_mean", "size_median", "size_std",
         "gap_to_size_mean",
         "rank_within_size_segment", "count_in_size_segment",
         "percentile_within_size_segment"]

out = df[keep].sort_values(["السنة", "القطاع", "فئة_SME", "rank_within_size_segment"])
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"Saved company vs size benchmarks to {OUTPUT}")