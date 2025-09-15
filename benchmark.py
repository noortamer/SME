import pandas as pd

# === Load Data ===
file_path = r"data/مؤشرات_الشركة_سنوياً_ar.csv"
df = pd.read_csv(file_path)

# === Sector Weights (7 features per sector, normalized to 1) ===
weights = {
    "الزراعة والحراجة وصيد الأسماك": [0.12, 0.10, 0.20, 0.08, 0.20, 0.20, 0.10],
    "التعدين واستغلال المحاجر": [0.18, 0.10, 0.12, 0.06, 0.18, 0.28, 0.08],
    "الصناعة التحويلية": [0.20, 0.12, 0.14, 0.08, 0.12, 0.26, 0.08],
    "إمدادات الكهرباء والغاز والبخار وتكييف الهواء": [0.18, 0.08, 0.12, 0.06, 0.18, 0.30, 0.08],
    "إمدادات المياه؛ الصرف الصحي وإدارة النفايات ومعالجتها": [0.16, 0.08, 0.12, 0.06, 0.18, 0.32, 0.08],
    "التشييد والبناء": [0.20, 0.14, 0.16, 0.10, 0.12, 0.20, 0.08],
    "تجارة الجملة والتجزئة؛ إصلاح المركبات ذات المحركات والدراجات النارية": [0.16, 0.18, 0.12, 0.12, 0.08, 0.14, 0.20],
    "النقل والتخزين": [0.18, 0.14, 0.16, 0.10, 0.10, 0.22, 0.10],
    "أنشطة الإقامة وخدمات الطعام": [0.14, 0.20, 0.12, 0.12, 0.08, 0.14, 0.20],
    "المعلومات والاتصالات": [0.12, 0.28, 0.10, 0.12, 0.06, 0.24, 0.08],
    "الأنشطة المالية وأنشطة التأمين": [0.16, 0.16, 0.12, 0.08, 0.12, 0.28, 0.08],
    "الأنشطة العقارية": [0.18, 0.10, 0.12, 0.06, 0.16, 0.30, 0.08],
    "الأنشطة المهنية والعلمية والتقنية": [0.14, 0.22, 0.10, 0.12, 0.08, 0.22, 0.12],
    "أنشطة الخدمات الإدارية وخدمات الدعم": [0.16, 0.18, 0.12, 0.12, 0.08, 0.18, 0.16],
    "الإدارة العامة والدفاع؛ الضمان الاجتماعي الإلزامي": [0.18, 0.06, 0.14, 0.06, 0.24, 0.24, 0.08],
    "التعليم": [0.14, 0.18, 0.12, 0.12, 0.16, 0.16, 0.12],
    "الصحة البشرية والعمل الاجتماعي": [0.14, 0.22, 0.12, 0.12, 0.12, 0.20, 0.08],
    "الفنون والترفيه والتسلية": [0.12, 0.20, 0.12, 0.14, 0.10, 0.16, 0.16],
    "أنشطة الخدمات الأخرى": [0.14, 0.18, 0.12, 0.12, 0.10, 0.18, 0.16],
    "أنشطة الأسر المعيشية كأصحاب عمل؛ أنشطة إنتاج السلع والخدمات للاستخدام الخاص": [0.16, 0.08, 0.18, 0.10, 0.20, 0.20, 0.08],
    "أنشطة المنظمات والهيئات خارج الإقليم": [0.18, 0.06, 0.12, 0.06, 0.24, 0.26, 0.08],
}
# normalize
for s in weights:
    total = sum(weights[s])
    weights[s] = [w / total for w in weights[s]]

# === Features to normalize ===
features = [
    "المبيعات_جنيه", "الإيرادات_جنيه", "الموظفون",
    "رأس_المال_المدفوع_جنيه", "branches",
    "sme_growth", "company_age"
]

# Add company age & growth placeholder if missing
if "company_age" not in df.columns:
    df["company_age"] = df["السنة"] - df["سنة_البدء"]
if "sme_growth" not in df.columns:
    df["sme_growth"] = df.groupby("الرقم_الضريبي")["المبيعات_جنيه"].pct_change().fillna(0)

# === Normalize by year & sector ===
for col in features:
    df[f"{col}_norm"] = 0.0
    for (year, sector), group in df.groupby(["السنة","القطاع"]):
        min_val, max_val = group[col].min(), group[col].max()
        if max_val > min_val:
            df.loc[group.index, f"{col}_norm"] = (group[col]-min_val)/(max_val-min_val)

# === Score calculation ===
def calc_score(row):
    sector = row["القطاع"]
    if sector not in weights: return None
    w = weights[sector]
    cols = [f"{c}_norm" for c in features]
    score = sum(row[c]*wi for c,wi in zip(cols,w))
    return round(score*10,2)

df["score"] = df.apply(calc_score, axis=1)

# === Benchmarking ===
# Industry average per year
industry_avg = df.groupby(["القطاع","السنة"])["score"].mean().reset_index(name="industry_avg")

# Merge back into df
df = df.merge(industry_avg, on=["القطاع","السنة"], how="left")
df["vs_industry"] = df["score"] - df["industry_avg"]

# Company score growth vs last year
df["vs_last_year"] = df.groupby("الرقم_الضريبي")["score"].diff()

# Industry benchmarking table
industry_trends = df.groupby(["القطاع","السنة"])["score"].mean().unstack()

# === Save ===
df.to_csv("data\scores_with_benchmarks.csv", index=False, encoding="utf-8-sig")
industry_trends.to_csv("data\industry_trends.csv", encoding="utf-8-sig")

print("✅ Scoring and benchmarking done.")
print("Company-level file: data\\scores_with_benchmarks.csv")
print("Industry-level file: data\\industry_trends.csv")