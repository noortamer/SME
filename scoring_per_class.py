# score_by_class.py
import pandas as pd
import numpy as np

DYN_PATH   = r"data\مؤشرات_الشركة_سنوياً_ar.csv"   
STATIC_PATH= r"data\الشركات_ar_enriched_fixed.csv"               
OUT_PATH   = r"data\finalscores_class.csv"
MIN_CLASS  = 10  


weights = {
    "الزراعة والحراجة وصيد الأسماك": [0.12,0.10,0.20,0.08,0.20,0.20,0.10],
    "التعدين واستغلال المحاجر":       [0.18,0.10,0.12,0.06,0.18,0.28,0.08],
    "الصناعة التحويلية":              [0.20,0.12,0.14,0.08,0.12,0.26,0.08],
    "إمدادات الكهرباء والغاز والبخار وتكييف الهواء":[0.18,0.08,0.12,0.06,0.18,0.30,0.08],
    "إمدادات المياه؛ الصرف الصحي وإدارة النفايات ومعالجتها":[0.16,0.08,0.12,0.06,0.18,0.32,0.08],
    "التشييد والبناء":               [0.20,0.14,0.16,0.10,0.12,0.20,0.08],
    "تجارة الجملة والتجزئة؛ إصلاح المركبات ذات المحركات والدراجات النارية":[0.16,0.18,0.12,0.12,0.08,0.14,0.20],
    "النقل والتخزين":                [0.18,0.14,0.16,0.10,0.10,0.22,0.10],
    "أنشطة الإقامة وخدمات الطعام":   [0.14,0.20,0.12,0.12,0.08,0.14,0.20],
    "المعلومات والاتصالات":           [0.12,0.28,0.10,0.12,0.06,0.24,0.08],
    "الأنشطة المالية وأنشطة التأمين": [0.16,0.16,0.12,0.08,0.12,0.28,0.08],
    "الأنشطة العقارية":               [0.18,0.10,0.12,0.06,0.16,0.30,0.08],
    "الأنشطة المهنية والعلمية والتقنية":[0.14,0.22,0.10,0.12,0.08,0.22,0.12],
    "أنشطة الخدمات الإدارية وخدمات الدعم":[0.16,0.18,0.12,0.12,0.08,0.18,0.16],
    "الإدارة العامة والدفاع؛ الضمان الاجتماعي الإلزامي":[0.18,0.06,0.14,0.06,0.24,0.24,0.08],
    "التعليم":                         [0.14,0.18,0.12,0.12,0.16,0.16,0.12],
    "الصحة البشرية والعمل الاجتماعي":  [0.14,0.22,0.12,0.12,0.12,0.20,0.08],
    "الفنون والترفيه والتسلية":       [0.12,0.20,0.12,0.14,0.10,0.16,0.16],
    "أنشطة الخدمات الأخرى":           [0.14,0.18,0.12,0.12,0.10,0.18,0.16],
    "أنشطة الأسر المعيشية كأصحاب عمل؛ أنشطة إنتاج السلع والخدمات للاستخدام الخاص":[0.16,0.08,0.18,0.10,0.20,0.20,0.08],
    "أنشطة المنظمات والهيئات خارج الإقليم":[0.18,0.06,0.12,0.06,0.24,0.26,0.08],
}
weights = {k: np.array(v)/sum(v) for k,v in weights.items()}

features = [
    "المبيعات_جنيه",      # Sales
    "نمو_المبيعات",       # Sales growth (YoY)
    "الموظفون",            # Employees
    "نمو_الموظفين",       # Employee growth (YoY)
    "عمر_المنشأة",         # Firm age (years)
    "العائد_على_رأس_المال",# Revenue/Capital
    "branches"             # Branches
]


df = pd.read_csv(DYN_PATH, encoding="utf-8-sig")
static = pd.read_csv(STATIC_PATH, encoding="utf-8-sig")

df.columns = df.columns.str.strip()
static.columns = static.columns.str.strip()

# unify branch column name if needed
if "الفروع" in df.columns and "branches" not in df.columns:
    df = df.rename(columns={"الفروع":"branches"})
if "branches" not in df.columns:
    df["branches"] = 0

# merge start_year & (optional) company name
keep_cols = ["الرقم_الضريبي","start_year","اسم_الشركة"]
have_cols = [c for c in keep_cols if c in static.columns]
df = df.merge(static[have_cols], on="الرقم_الضريبي", how="left")

# ---------------- Build derived features ----------------
# Age
df["start_year"] = df["start_year"].fillna(df["السنة"])
df["عمر_المنشأة"] = (df["السنة"] - df["start_year"]).clip(lower=0) + 1

# Sort for growth
df = df.sort_values(["الرقم_الضريبي","السنة"]).reset_index(drop=True)

# Growth (keep NaN for first year, we’ll median-impute inside percentile function)
df["نمو_المبيعات"]   = df.groupby("الرقم_الضريبي")["المبيعات_جنيه"].pct_change()
df["نمو_الموظفين"]   = df.groupby("الرقم_الضريبي")["الموظفون"].pct_change()

# Revenue / Paid capital (avoid div-by-zero)
den = df["رأس_المال_المدفوع_جنيه"].replace(0, np.nan)
df["العائد_على_رأس_المال"] = df["الإيرادات_جنيه"] / den
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------- Percentile helper ----------------
def pct_in_group(s: pd.Series) -> pd.Series:
    # neutral-impute NaNs with group median for ranking only
    s2 = s.copy()
    if s2.isna().all():
        return pd.Series(0.5, index=s.index)
    s2 = s2.fillna(s2.median())
    if s2.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=s.index)
    return s2.rank(pct=True, method="average")

# ---------------- Class-aware percentiles with fallback ----------------
# Compute class-year sizes
if "الفرع_كود" not in df.columns:
    raise ValueError("Column 'الفرع_كود' (ISIC class code) is missing in the dynamic dataset.")

class_sizes = df.groupby(["السنة","القطاع","الفرع_كود"])["الرقم_الضريبي"].transform("nunique")
use_class = class_sizes >= MIN_CLASS

# Prepare containers
for col in features:
    df[col+"_p_class"]  = np.nan
    df[col+"_p_sector"] = np.nan

# Class-level percentiles (where group big enough)
for col in features:
    df[col+"_p_class"] = df.groupby(["السنة","القطاع","الفرع_كود"], dropna=False)[col].transform(pct_in_group)

# Sector-level percentiles (fallback)
for col in features:
    df[col+"_p_sector"] = df.groupby(["السنة","القطاع"], dropna=False)[col].transform(pct_in_group)

# Pick which to use
for col in features:
    df[col+"_p"] = np.where(use_class, df[col+"_p_class"], df[col+"_p_sector"])

# Record what we used & how many peers
df["group_used"] = np.where(use_class, "class", "sector")
df["group_size_used"] = np.where(use_class,
                                 df.groupby(["السنة","القطاع","الفرع_كود"])["الرقم_الضريبي"].transform("nunique"),
                                 df.groupby(["السنة","القطاع"])["الرقم_الضريبي"].transform("nunique")).astype(int)

# ---------------- Weighted score (0–10) ----------------
def row_score(row):
    sec = row["القطاع"]
    w = weights.get(sec, np.ones(len(features))/len(features))
    pvals = row[[f"{c}_p" for c in features]].to_numpy(dtype=float)
    return float(np.dot(pvals, w) * 10.0)

df["score"] = df.apply(row_score, axis=1).round(2)

# ---------------- Save ----------------
# Optional: drop the helper _class/_sector columns to keep file light
drop_helpers = []
for c in features:
    drop_helpers += [f"{c}_p_class", f"{c}_p_sector"]

out = df.drop(columns=drop_helpers, errors="ignore")
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

# ---------------- Console summary ----------------
print(f"Saved class-aware scores to {OUT_PATH}")
print("Group usage share:")
print(out["group_used"].value_counts())
print("\nExample rows:")
cols_show = ["الرقم_الضريبي","اسم_الشركة","السنة","القطاع","الفرع_كود","group_used","group_size_used","score"]
print(out[cols_show].head(10))