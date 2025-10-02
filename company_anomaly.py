# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

IN_PATH  = r"data\new_scores_with_inflation.csv"   # CPI-adjusted file
OUT_PATH = r"data\new_anomaly_detection_results.csv"
MIN_CLASS = 10

SECTOR_ICT         = "المعلومات والاتصالات"
SECTOR_AGRICULTURE = "الزراعة والحراجة وصيد الأسماك"
SECTOR_MINING      = "التعدين واستغلال المحاجر"

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

need = ["الرقم_الضريبي","السنة","القطاع",
        "المبيعات_حقيقي","النمو_الحقيقي_للمبيعات",
        "الموظفون","رأس_المال_المدفوع_جنيه","العائد_على_رأس_المال"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Numerics
for c in ["السنة","المبيعات_حقيقي","النمو_الحقيقي_للمبيعات",
          "الموظفون","رأس_المال_المدفوع_جنيه","العائد_على_رأس_المال"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Avoid division by zero artifacts
df["الموظفون"] = df["الموظفون"].replace(0, np.nan)
df["رأس_المال_المدفوع_جنيه"] = df["رأس_المال_المدفوع_جنيه"].replace(0, np.nan)

# Real Sales per Employee
df["مبيعات_لكل_موظف"] = df["المبيعات_حقيقي"] / df["الموظفون"]

# Choose peer group: (year, sector, class) if big enough; else (year, sector)
has_class = "الفرع_كود" in df.columns
if has_class:
    class_counts = df.groupby(["السنة","القطاع","الفرع_كود"])["الرقم_الضريبي"].transform("nunique")
    use_class = class_counts >= MIN_CLASS
else:
    use_class = pd.Series(False, index=df.index)

def winsorize_in_group(series, lower=0.01, upper=0.99):
    ql = series.quantile(lower)
    qu = series.quantile(upper)
    return series.clip(ql, qu)

def transform_group(col, fn, by_class=True):
    if has_class and by_class:
        v_class  = df.groupby(["السنة","القطاع","الفرع_كود"], dropna=False)[col].transform(fn)
        v_sector = df.groupby(["السنة","القطاع"], dropna=False)[col].transform(fn)
        out = np.where(use_class, v_class, v_sector)
        return pd.Series(out, index=df.index)
    else:
        return df.groupby(["السنة","القطاع"], dropna=False)[col].transform(fn)

# Winsorized series for stable thresholds
for metric in ["المبيعات_حقيقي","مبيعات_لكل_موظف","الموظفون",
               "رأس_المال_المدفوع_جنيه","العائد_على_رأس_المال"]:
    df[f"{metric}__w"] = transform_group(metric, lambda s: winsorize_in_group(s), by_class=True)

# Z-score on real sales (winsorized)
mu = transform_group("المبيعات_حقيقي__w", "mean", by_class=True)
sd = transform_group("المبيعات_حقيقي__w", "std",  by_class=True)
z = (df["المبيعات_حقيقي__w"] - mu) / sd.replace(0, np.nan)
df["z_within_industry_year"] = z.fillna(0.0)
df["z_valid"] = (~sd.isna() & (sd > 0)).astype(int)  # audit

# Percentile thresholds on winsorized series
def q(pct): return pct/100.0
P_DEF = {
    "مبيعات_لكل_موظف__w": [10, 20, 97.5, 99, 99.5],
    "الموظفون__w":         [10, 20, 90, 95],
    "رأس_المال_المدفوع_جنيه__w": [90, 95],
    "العائد_على_رأس_المال__w":  [5, 10, 15, 25],
}
for col, plist in P_DEF.items():
    for p in plist:
        colp = f"{col}_p{p}"
        if has_class:
            v_class  = df.groupby(["السنة","القطاع","الفرع_كود"], dropna=False)[col].transform("quantile", q(p))
            v_sector = df.groupby(["السنة","القطاع"], dropna=False)[col].transform("quantile", q(p))
            df[colp] = np.where(use_class, v_class, v_sector)
        else:
            df[colp] = df.groupby(["السنة","القطاع"], dropna=False)[col].transform("quantile", q(p))

# ---------------- RULES (same business logic) ----------------
SECT = df["القطاع"]

# R1: Sales spike
r1_growth = np.where(SECT.eq(SECTOR_ICT), 6.0,
             np.where(SECT.eq(SECTOR_AGRICULTURE), 2.5,
             np.where(SECT.eq(SECTOR_MINING), 5.0, 4.0)))
r1_z = np.where(SECT.eq(SECTOR_ICT), 3.5,
        np.where(SECT.eq(SECTOR_AGRICULTURE), 2.5,
        np.where(SECT.eq(SECTOR_MINING), 3.0, 3.0)))
R1 = (
    (df["النمو_الحقيقي_للمبيعات"] >= r1_growth) &
    (df["ز_within_industry_year" if False else "z_within_industry_year"] >= r1_z) &
    (pd.to_numeric(df.get("نمو_الموظفين", np.nan), errors="coerce") <= 0.20)
).astype(int)

# R2: High sales + low employees
r2_hi = np.where(SECT.eq(SECTOR_ICT),
                 df["مبيعات_لكل_موظف__w_p99.5"],
                 df["مبيعات_لكل_موظف__w_p99"])
R2 = (
    (df["مبيعات_لكل_موظف"] >= r2_hi) &
    (df["الموظفون"]        <= df["الموظفون__w_p10"]) &
    (df["z_within_industry_year"] >= 2.5)
).astype(int)

# R3: High employees + low sales
R3 = (
    (df["الموظفون"]        >= df["الموظفون__w_p95"]) &
    (df["مبيعات_لكل_موظف"] <= df["مبيعات_لكل_موظف__w_p10"]) &
    (df["النمو_الحقيقي_للمبيعات"] <= 0.10)
).astype(int)

# R4: High capital + poor utilization (sector-specific ROC threshold only)
r4_roc_low = np.where(SECT.eq(SECTOR_MINING),
                      df["العائد_على_رأس_المال__w_p5"],
                      df["العائد_على_رأس_المال__w_p10"])
R4 = (
    (df["رأس_المال_المدفوع_جنيه"] >= df["رأس_المال_المدفوع_جنيه__w_p95"]) &
    (df["العائد_على_رأس_المال"]   <= r4_roc_low) &
    (df["النمو_الحقيقي_للمبيعات"].abs() <= 0.05)
).astype(int)

# Sensitive variants
s1_growth = np.where(SECT.eq(SECTOR_ICT), 3.5,
             np.where(SECT.eq(SECTOR_AGRICULTURE), 1.5,
             np.where(SECT.eq(SECTOR_MINING), 3.0, 2.0)))
s1_z = np.where(SECT.eq(SECTOR_ICT), 2.5,
         np.where(SECT.eq(SECTOR_AGRICULTURE), 1.5,
         np.where(SECT.eq(SECTOR_MINING), 2.0, 2.0)))
S1 = (
    (df["النمو_الحقيقي_للمبيعات"] >= s1_growth) &
    (df["z_within_industry_year"]  >= s1_z) &
    (pd.to_numeric(df.get("نمو_الموظفين", np.nan), errors="coerce") <= 0.30)
).astype(int)

s2_hi = np.where(SECT.eq(SECTOR_ICT),
                 df["مبيعات_لكل_موظف__w_p99"],
                 df["مبيعات_لكل_موظف__w_p97.5"])
S2 = (
    (df["مبيعات_لكل_موظف"] >= s2_hi) &
    (df["الموظفون"]        <= df["الموظفون__w_p20"]) &
    (df["z_within_industry_year"] >= 2.0)
).astype(int)

S3 = (
    (df["الموظفون"]        >= df["الموظفون__w_p90"]) &
    (df["مبيعات_لكل_موظف"] <= df["مبيعات_لكل_موظف__w_p20"])
).astype(int)

s4_roc_low = np.where(SECT.eq(SECTOR_MINING),
                      df["العائد_على_رأس_المال__w_p15"],
                      df["العائد_على_رأس_المال__w_p25"])
S4 = (
    (df["رأس_المال_المدفوع_جنيه"] >= df["رأس_المال_المدفوع_جنيه__w_p90"]) &
    (df["العائد_على_رأس_المال"]   <= s4_roc_low) &
    (df["النمو_الحقيقي_للمبيعات"].abs() <= 0.10)
).astype(int)

# Collect rule flags
df["R1"], df["R2"], df["R3"], df["R4"] = R1, R2, R3, R4
df["S1"], df["S2"], df["S3"], df["S4"] = S1, S2, S3, S4
df["conservative_anomaly"] = (df[["R1","R2","R3","R4"]].sum(axis=1) > 0).astype(int)
df["sensitive_anomaly"]    = (df[["S1","S2","S3","S4"]].sum(axis=1) > 0).astype(int)
df["عدد_القواعد_R"] = df[["R1","R2","R3","R4"]].sum(axis=1)
df["عدد_القواعد_S"] = df[["S1","S2","S3","S4"]].sum(axis=1)

# ---------------- NEW: Multi-year consistency flag ----------------
# Flag if the same company trips any rule in >= 2 consecutive years
df = df.sort_values(["الرقم_الضريبي","السنة"])
any_rule = (
    (df[["R1","R2","R3","R4","S1","S2","S3","S4"]].sum(axis=1) > 0).astype(int)
)
# consecutive-year trigger per company
consecutive = any_rule.groupby(df["الرقم_الضريبي"]).apply(lambda s: (s.shift(1) == 1) & (s == 1)).reset_index(level=0, drop=True)
# Multi-year consistency if ever had a consecutive pair
df["multi_year_consistency"] = consecutive.groupby(df["الرقم_الضريبي"]).transform(lambda s: 1 if s.any() else 0).astype(int)

# ---------------- Output ----------------
base_cols = [
    "الرقم_الضريبي","اسم_الشركة","السنة","القطاع",
    "القسم_كود","المجموع_كود","الفرع_كود",
    "فئة_SME","سنة_البداية","عمر_المنشأة",
    "المبيعات_حقيقي","النمو_الحقيقي_للمبيعات",
    "الموظفون","رأس_المال_المدفوع_جنيه","العائد_على_رأس_المال",
    "مبيعات_لكل_موظف","z_within_industry_year","z_valid","score"
]
base_cols = [c for c in base_cols if c in df.columns]

th_cols = [
    "مبيعات_لكل_موظف__w_p10","مبيعات_لكل_موظف__w_p20","مبيعات_لكل_موظف__w_p97.5","مبيعات_لكل_موظف__w_p99","مبيعات_لكل_موظف__w_p99.5",
    "الموظفون__w_p10","الموظفون__w_p20","الموظفون__w_p90","الموظفون__w_p95",
    "رأس_المال_المدفوع_جنيه__w_p90","رأس_المال_المدفوع_جنيه__w_p95",
    "العائد_على_رأس_المال__w_p5","العائد_على_رأس_المال__w_p10","العائد_على_رأس_المال__w_p15","العائد_على_رأس_المال__w_p25"
]

rule_cols = ["R1","R2","R3","R4","S1","S2","S3","S4",
             "conservative_anomaly","sensitive_anomaly",
             "عدد_القواعد_R","عدد_القواعد_S","multi_year_consistency"]

out = df[base_cols + th_cols + rule_cols]
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

# Diagnostics
print("Anomaly detection completed!")
print(f"Total rows: {len(out)}")
print(f"Rows with conservative anomalies: {int(out['conservative_anomaly'].sum())}")
print(f"Rows with sensitive anomalies: {int(out['sensitive_anomaly'].sum())}")
print(f"Rows with multi-year consistency: {int(out['multi_year_consistency'].sum())}")
print("Zero-variance groups (z_valid=0) rows:", int((df['z_valid']==0).sum()))
print("Peer sizes (class>=MIN_CLASS) share:", round(float(use_class.mean())*100,2) if has_class else "No class column")