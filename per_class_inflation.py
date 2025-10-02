import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = r"data\finalscores_class.csv"   # change if needed
OUT_PATH = r"data\new_scores_with_inflation.csv"

# Annual inflation (your figures)
INFLATION = {
    2020: 0.07168,
    2021: 0.04285,
    2022: 0.07259,
    2023: 0.31932,
    2024: 0.35710,
    2025: 0.23950,
}

# -------- Load & basic checks --------
df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

need = ["الرقم_الضريبي", "السنة", "المبيعات_جنيه"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Coerce numerics (robustness)
df["السنة"] = pd.to_numeric(df["السنة"], errors="coerce")
df["المبيعات_جنيه"] = pd.to_numeric(df["المبيعات_جنيه"], errors="coerce")
if "نمو_المبيعات" in df.columns:
    df["نمو_المبيعات"] = pd.to_numeric(df["نمو_المبيعات"], errors="coerce")

# Sort AFTER coercion so pct_change is correct
df = df.sort_values(["الرقم_الضريبي", "السنة"]).reset_index(drop=True)

# -------- Build CPI chain --------
years = df["السنة"].dropna().astype(int).unique().tolist()
overlap = [y for y in years if y in INFLATION]

if not overlap:
    # No overlap between data years and inflation table → still add empty cols
    df["نسبة_التضخم"] = np.nan
    df["CPI_index"] = np.nan
    df["المبيعات_حقيقي"] = np.nan
    df["النمو_الحقيقي_للمبيعات"] = np.nan
else:
    base_year = max(overlap)  # latest year with inflation info
    cpi = {base_year: 1.0}
    # back-fill older years
    for y in sorted([yy for yy in years if yy < base_year], reverse=True):
        cpi[y] = cpi.get(y+1, np.nan) / (1.0 + INFLATION.get(y+1, 0.0))
    # forward-fill newer years
    for y in sorted([yy for yy in years if yy > base_year]):
        cpi[y] = cpi.get(y-1, np.nan) * (1.0 + INFLATION.get(y, 0.0))

    df["نسبة_التضخم"] = df["السنة"].map(INFLATION)
    df["CPI_index"]   = df["السنة"].map(cpi)

    # Guard (very unlikely)
    bad = (df["CPI_index"] <= 0) | ~np.isfinite(df["CPI_index"])
    df.loc[bad, "CPI_index"] = np.nan

    # Real sales & real YoY growth
    df["المبيعات_حقيقي"] = df["المبيعات_جنيه"] / df["CPI_index"]
    df["النمو_الحقيقي_للمبيعات"] = (
        df.groupby("الرقم_الضريبي")["المبيعات_حقيقي"].pct_change()
    )

    # Fallback: if real growth is NaN but nominal exists, copy nominal
    if "نمو_المبيعات" in df.columns:
        mask_nom = df["النمو_الحقيقي_للمبيعات"].isna() & df["نمو_المبيعات"].notna()
        df.loc[mask_nom, "النمو_الحقيقي_للمبيعات"] = df.loc[mask_nom, "نمو_المبيعات"]

# -------- Reorder for readability (optional) --------
cols = df.columns.tolist()

def move_after(lst, col, anchor):
    if col in lst and anchor in lst:
        lst.insert(lst.index(anchor) + 1, lst.pop(lst.index(col)))

# Place inflation/CPI after year
move_after(cols, "نسبة_التضخم", "السنة")
move_after(cols, "CPI_index", "السنة")

# Place real sales next to nominal sales
move_after(cols, "المبيعات_حقيقي", "المبيعات_جنيه")

# Place real growth next to nominal growth (if nominal exists)
if "نمو_المبيعات" in cols:
    move_after(cols, "النمو_الحقيقي_للمبيعات", "نمو_المبيعات")

df = df[cols]

# Light rounding for readability (does NOT touch your score)
for c in ["نسبة_التضخم", "CPI_index", "المبيعات_حقيقي", "النمو_الحقيقي_للمبيعات"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(6)

# -------- Save --------
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved → {OUT_PATH}")