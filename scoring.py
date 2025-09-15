# === Simple Scoring for Arabic dataset (no VAT, tier-aware growth) ===
# Input : مؤشرات_الشركة_سنوياً_ar.csv   (your Arabic metrics file)
#         الشركات_ar.csv                 (used only to fetch start_year if missing)
# Output: scores_ar.csv

import pandas as pd
import numpy as np
import os

# ---- paths (EDIT if needed) ----
data_dir = r"C:\Users\na255073\Documents\SME_dataset\new_data"
metrics_path  = os.path.join(data_dir, "مؤشرات_الشركة_سنوياً_ar.csv")
firms_path    = os.path.join(data_dir, "الشركات_ar.csv")
out_path      = os.path.join(data_dir, "scores_ar.csv")

# ---- load ----
df = pd.read_csv(metrics_path, encoding="utf-8-sig")
# If start_year is not in metrics, try to merge it from companies file
if "start_year" not in df.columns and os.path.exists(firms_path):
    comp = pd.read_csv(firms_path, encoding="utf-8-sig")
    if "start_year" in comp.columns:
        df = df.merge(comp[["الرقم_الضريبي","start_year"]], on="الرقم_الضريبي", how="left")

# ---- 1) build derived features ----
df = df.sort_values(["الرقم_الضريبي","السنة"])

# Sales growth (YOY) per company
df["نمو_المبيعات"] = (
    df.groupby("الرقم_الضريبي")["المبيعات_جنيه"]
      .pct_change()
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0.0)
)

# Age (prefer explicit start_year; else infer from first year seen)
if "start_year" in df.columns and df["start_year"].notna().any():
    df["العمر"] = df["السنة"] - df["start_year"]
else:
    first_seen = df.groupby("الرقم_الضريبي")["السنة"].transform("min")
    df["العمر"] = df["السنة"] - first_seen

# Revenue / Capital
den = df["رأس_المال_المدفوع_جنيه"].replace(0, np.nan)
df["نسبة_الإيرادات_لرأس_المال"] = (df["الإيرادات_جنيه"] / den).fillna(0.0)

# Branches column name is already "branches" in your AR metrics per your last script
if "branches" not in df.columns:
    # safe fallback if missing: set to 1
    df["branches"] = 1

# SME tier as numeric (small=0, medium=1)
def tier_num(x):
    # Works with Arabic ("صغير","متوسط") or English ("small","medium")
    s = str(x)
    if "medium" in s or "متوسط" in s:
        return 1.0
    if "small" in s or "صغير" in s:
        return 0.0
    return 0.5  # unknown -> neutral
df["SME_num"] = df.get("فئة_SME", "").apply(tier_num)

# ---- 2) normalize features (sector,year) ----
# MinMax helper
def minmax(g):
    lo, hi = g.min(), g.max()
    return (g - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=g.index)

grp = df.groupby(["القطاع","السنة"], group_keys=False)

# a) Sales level
df["n_sales"]  = grp["المبيعات_جنيه"].apply(minmax)
# b) Employees
df["n_emp"]    = grp["الموظفون"].apply(minmax)
# c) Age
df["n_age"]    = grp["العمر"].apply(minmax)
# d) Rev/Cap ratio
df["n_rc"]     = grp["نسبة_الإيرادات_لرأس_المال"].apply(minmax)
# e) Branches
df["n_branch"] = grp["branches"].apply(minmax)
# f) SME tier (category -> keep as is; no min-max)
df["n_sme"]    = df["SME_num"]

# g) Sales growth (tier-aware scaling, not min-max)
# thresholds: small gets "full" credit at ~30% growth; medium at ~15%
small_full, med_full = 0.30, 0.15
def tier_growth_score(row):
    thr = med_full if row["SME_num"] >= 0.5 else small_full
    return float(np.clip(row["نمو_المبيعات"] / thr, 0, 1))
df["n_growth"] = df.apply(tier_growth_score, axis=1)

# ---- 3) sector-specific weights (7 numbers in this exact order) ----
# order: [sales_level, sales_growth, employees, age, rev_cap_ratio, branches, sme_tier]
weights = {
    # EXAMPLE: put your sector Arabic names here with the 7 weights.
    # Make sure they sum to ~1; we’ll normalize just in case.
    "المعلومات والاتصالات": [0.15, 0.25, 0.10, 0.05, 0.25, 0.10, 0.10],
    "الصناعة التحويلية":     [0.20, 0.15, 0.15, 0.10, 0.15, 0.10, 0.15],
    # ... add the rest of your 21 sectors here ...
}
# Fallback: equal weights if a sector isn’t listed above
default_w = np.array([1/7]*7, dtype=float)

# Normalize all weights to sum=1
for s in list(weights.keys()):
    w = np.array(weights[s], dtype=float)
    tot = w.sum()
    weights[s] = (w / tot).tolist() if tot > 0 else default_w.tolist()

# ---- 4) compute score (0–10) ----
feat_cols = ["n_sales","n_growth","n_emp","n_age","n_rc","n_branch","n_sme"]

def row_score(r):
    w = weights.get(r["القطاع"], default_w)
    return round(float(np.dot(r[feat_cols].values, np.array(w))) * 10, 2)

df["score"] = df.apply(row_score, axis=1)

# ---- 5) save ----
keep = ["الرقم_الضريبي","السنة","القطاع","الفرع_كود","القسم_كود","المجموع_كود",
        "المبيعات_جنيه","الإيرادات_جنيه","الموظفون","رأس_المال_المدفوع_جنيه",
        "branches","فئة_SME","نمو_المبيعات","العمر","نسبة_الإيرادات_لرأس_المال","score"]
df[keep].to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)