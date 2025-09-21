import pandas as pd
import numpy as np

file_path = r"data\مؤشرات_الشركة_سنوياً_ar.csv"
df = pd.read_csv(file_path)

companies_file_path = r"data\الشركات_ar.csv"
companies_df = pd.read_csv(companies_file_path)

df.rename(columns={'branches': 'الفروع'}, inplace=True)

company_info = companies_df[['الرقم_الضريبي', 'start_year']]
df = pd.merge(df, company_info, on='الرقم_الضريبي', how='left')


df['start_year'].fillna(df['السنة'], inplace=True)
df['عمر_المنشأة'] = (df['السنة'] - df['start_year']) + 1
df['عمر_المنشأة'] = df['عمر_المنشأة'].apply(lambda x: max(x, 1))
df.drop(columns=['start_year'], inplace=True)

df = df.sort_values(by=['الرقم_الضريبي', 'السنة']).reset_index(drop=True)

df['نمو_المبيعات'] = df.groupby('الرقم_الضريبي')['المبيعات_جنيه'].pct_change()
df['نمو_الموظفين'] = df.groupby('الرقم_الضريبي')['الموظفون'].pct_change()

denominator = df['رأس_المال_المدفوع_جنيه'].replace(0, np.nan)
df['العائد_على_رأس_المال'] = df['الإيرادات_جنيه'] / denominator

df['نمو_المبيعات'].fillna(0, inplace=True)
df['نمو_الموظفين'].fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)
df['العائد_على_رأس_المال'].fillna(0, inplace=True)



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

#normalize the weights to ensure their sum = 1
for sector in weights:
    total_weight = sum(weights[sector])
    if total_weight > 0:
        weights[sector] = [w / total_weight for w in weights[sector]]

# List of columns to normalize, with all Arabic names
# Corresponds to: SL, SG, E, EG, AGE, RC, BR
columns_to_normalize = [
    "المبيعات_جنيه",
    "نمو_المبيعات",
    "الموظفون",
    "نمو_الموظفين",
    "عمر_المنشأة",
    "العائد_على_رأس_المال",
    "الفروع",
]


for col in columns_to_normalize:
    df[f"{col}_normalized"] = 0.0

size_column = 'فئة_SME'

#normalize data within each year AND each company size group AND each sector
for year in df['السنة'].unique():
    for size in df[size_column].unique():
        for sector in df['القطاع'].unique():
            #create a boolean mask for the current year, company size segment, and sector
            mask = (df['السنة'] == year) & (df[size_column] == size) & (df['القطاع'] == sector)
            
            #check if the filtered segment is not empty
            if not df.loc[mask].empty:
                for col in columns_to_normalize:
                    min_val = df.loc[mask, col].min()
                    max_val = df.loc[mask, col].max()
                    range_val = max_val - min_val
                    
                    if range_val > 0:
                        #apply min-max normalization
                        df.loc[mask, f"{col}_normalized"] = (df.loc[mask, col] - min_val) / range_val
                    else:
                        #all values in the segment are the same then value = 0
                        df.loc[mask, f"{col}_normalized"] = 0

#calculate the final weighted score
def calculate_normalized_score(row):
    sector = row["القطاع"]
    if sector in weights:
        sector_weights = weights[sector]
        score = 0
        normalized_cols = [f"{c}_normalized" for c in columns_to_normalize]
        for col, weight in zip(normalized_cols, sector_weights):
            score += row[col] * weight
        return round(score * 10, 2)
    return None

df["score"] = df.apply(calculate_normalized_score, axis=1)



output_path = r"data\scores_peers.csv"
normalized_cols_to_drop = [f"{c}_normalized" for c in columns_to_normalize]
output_df = df.drop(columns=normalized_cols_to_drop)

numeric_cols = output_df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    output_df[col] = output_df[col].round(2)

output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Scores calculated correctly using segmented normalization and saved to {output_path}")
print("\nScore statistics:")
print(df['score'].describe())