# company_anomalies.py  —  robust Step 4 implementation
import pandas as pd
import numpy as np

# ---------------- I/O ----------------
INPUT_YEARLY = r"data/مؤشرات_الشركة_سنوياً_ar.csv"   # yearly panel
INPUT_COMPANIES = r"data/الشركات_ar.csv"             # master (for names)
OUTPUT = r"data/company_anomalies.csv"

# ---------------- Config (tunable) ----------------
MIN_GROUP = 8  # need at least this many peers to use size-split; otherwise fall back to sector×year

# Sector-specific thresholds (as agreed)
SECTOR_RULES = {
    "المعلومات والاتصالات": {"yoy_abs": 2.0, "yoy_pct": 0.95, "rc_lo": 0.10},
    "الأنشطة المالية وأنشطة التأمين": {"yoy_abs": 2.0, "yoy_pct": 0.95, "rc_lo": 0.10},
    "الصناعة التحويلية": {"yoy_abs": 0.30, "yoy_pct": 0.90, "rc_lo": 0.15},
    "التشييد والبناء": {"yoy_abs": 0.30, "yoy_pct": 0.90, "rc_lo": 0.15},
    "تجارة الجملة والتجزئة؛ إصلاح المركبات ذات المحركات والدراجات النارية": {"yoy_abs": 3.0, "yoy_pct": 0.99, "rc_lo": 0.10},
    "الزراعة والحراجة وصيد الأسماك": {"yoy_abs": 3.0, "yoy_pct": 0.99, "rc_lo": 0.10},
    "أنشطة الإقامة وخدمات الطعام": {"yoy_abs": 2.0, "yoy_pct": 0.95, "rc_lo": 0.10},
    "_default": {"yoy_abs": 2.0, "yoy_pct": 0.95, "rc_lo": 0.10},
}

def rules_for(sec: str) -> dict:
    return SECTOR_RULES.get(sec, SECTOR_RULES["_default"])

def to_num(x):
    """robust numeric cast: remove commas/extra chars then to_numeric"""
    s = pd.Series(x).astype(str).str.replace(r"[^\d\.-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def pct_rank(series: pd.Series) -> pd.Series:
    """percentile rank within group; neutral 0.5 if no variation"""
    s = series.copy()
    s = s.fillna(s.median())
    if s.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True, method="average")

# ---------------- Load & clean ----------------
df = pd.read_csv(INPUT_YEARLY, encoding="utf-8-sig")
df.columns = df.columns.str.strip()  # kill hidden spaces

# sanity check required columns
REQUIRED = ["الرقم_الضريبي", "السنة", "القطاع", "المبيعات_جنيه", "الإيرادات_جنيه", "الموظفون", "رأس_المال_المدفوع_جنيه"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nHave: {df.columns.tolist()}")

# numeric coercion
df["السنة"] = to_num(df["السنة"])
for c in ["المبيعات_جنيه","الإيرادات_جنيه","الموظفون","رأس_المال_المدفوع_جنيه"]:
    df[c] = to_num(df[c])

# drop rows without id/year/sales
df = df.dropna(subset=["الرقم_الضريبي","السنة","المبيعات_جنيه"]).copy()

# merge company names (optional but nice)
try:
    comp = pd.read_csv(INPUT_COMPANIES, encoding="utf-8-sig")
    comp.columns = comp.columns.str.strip()
    if "الرقم_الضريبي" in comp.columns and "اسم_الشركة" in comp.columns:
        df = df.merge(comp[["الرقم_الضريبي","اسم_الشركة"]], on="الرقم_الضريبي", how="left")
except FileNotFoundError:
    pass  # proceed without names

# ---------------- Aggregate duplicates per (id, year) BEFORE YoY ----------------
# sum sales & revenue; take first for categorical fields
agg_map = {
    "المبيعات_جنيه": "sum",
    "الإيرادات_جنيه": "sum",
    "الموظفون": "sum",  # if multiple branches reported separately, sum staff
    "رأس_المال_المدفوع_جنيه": "sum",
    "القطاع": "first",
    "فئة_SME": "first",
    "اسم_الشركة": "first",
}
# keep only columns present
agg_map = {k:v for k,v in agg_map.items() if k in df.columns}

df_agg = (
    df.groupby(["الرقم_الضريبي","السنة"], as_index=False)
      .agg(agg_map)
)

# ---------------- Compute YoY per company ----------------
df_agg = df_agg.sort_values(["الرقم_الضريبي","السنة"])
df_agg["sales_yoy"] = df_agg.groupby("الرقم_الضريبي")["المبيعات_جنيه"].pct_change()

# ---------------- Derive RC and prep for peer ranking ----------------
df_agg["rc"] = df_agg["الإيرادات_جنيه"] / df_agg["رأس_المال_المدفوع_جنيه"].replace(0, np.nan)
df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------- Percentiles vs peers ----------------
# Use size-aware grouping only if every size subgroup in (sector×year) has >= MIN_GROUP
use_size = "فئة_SME" in df_agg.columns

def group_keys_for_chunk(chunk: pd.DataFrame):
    if use_size:
        counts = chunk.groupby("فئة_SME")["المبيعات_جنيه"].count()
        if (counts >= MIN_GROUP).all():
            return ["القطاع","السنة","فئة_SME"]
    return ["القطاع","السنة"]

df_agg[["p_المبيعات_جنيه","p_الموظفون","p_rc","p_sales_yoy","group_n"]] = np.nan

for (sec, yr), chunk in df_agg.groupby(["القطاع","السنة"], dropna=False):
    keys = group_keys_for_chunk(chunk)
    for _, sub in chunk.groupby(keys, dropna=False):
        idx = sub.index
        df_agg.loc[idx, "group_n"] = len(sub)
        df_agg.loc[idx, "p_المبيعات_جنيه"] = pct_rank(sub["المبيعات_جنيه"])
        df_agg.loc[idx, "p_الموظفون"]     = pct_rank(sub["الموظفون"])
        df_agg.loc[idx, "p_rc"]            = pct_rank(sub["rc"])
        df_agg.loc[idx, "p_sales_yoy"]     = pct_rank(sub["sales_yoy"])

# ---------------- Apply anomaly rules ----------------
def apply_rules(row):
    r = rules_for(row["القطاع"])
    flag_spike = (pd.notna(row["p_sales_yoy"]) and row["p_sales_yoy"] >= r["yoy_pct"]) \
                 or (pd.notna(row["sales_yoy"]) and row["sales_yoy"] >= r["yoy_abs"])
    return pd.Series({
        "flag_sales_spike_yoy": bool(flag_spike),
        "flag_high_sales_low_emp": bool((row.get("p_المبيعات_جنيه",np.nan) >= 0.90) and (row.get("p_الموظفون",np.nan) <= 0.10)),
        "flag_high_emp_low_sales": bool((row.get("p_الموظفون",np.nan) >= 0.90) and (row.get("p_المبيعات_جنيه",np.nan) <= 0.10)),
        "flag_high_cap_stagnant": bool((row.get("p_rc",np.nan) <= r["rc_lo"]) and (pd.isna(row["sales_yoy"]) or row["sales_yoy"] <= 0)),
        "flag_low_rc": bool(row.get("p_rc",np.nan) <= r["rc_lo"]),
    })

flags = df_agg.apply(apply_rules, axis=1)
df_agg = pd.concat([df_agg, flags], axis=1)
flag_cols = [c for c in df_agg.columns if c.startswith("flag_")]
df_agg["anomaly_count"] = df_agg[flag_cols].sum(axis=1)

# ---------------- Output ----------------
keep = ["الرقم_الضريبي","اسم_الشركة","السنة","القطاع","فئة_SME",
        "المبيعات_جنيه","الإيرادات_جنيه","الموظفون","رأس_المال_المدفوع_جنيه",
        "sales_yoy","rc","group_n","anomaly_count"] + flag_cols
keep = [c for c in keep if c in df_agg.columns]

out = df_agg[keep].sort_values(["السنة","القطاع","anomaly_count"], ascending=[True, True, False])
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

# Quick visibility
print(f"[OK] Saved company anomalies to {OUTPUT}")
print("Sample:")
print(out.head(10))