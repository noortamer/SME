import pandas as pd
import numpy as np

df = pd.read_csv("data/scores_after_inflation.csv")

#sector exceptions
SECTOR_ICT = "المعلومات والاتصالات"
SECTOR_AGRICULTURE = "الزراعة والحراجة وصيد الأسماك"
SECTOR_MINING = "التعدين واستغلال المحاجر"

# NaN values in comparisons will evaluate to False, preventing false positives
df['الموظفون'] = df['الموظفون'].replace(0, np.nan)
df['رأس_المال_المدفوع_جنيه'] = df['رأس_المال_المدفوع_جنيه'].replace(0, np.nan)

df['مبيعات_لكل_موظف'] = df['المبيعات_بعد_التضخم'] / df['الموظفون']

#z-score within industry-year groups
df['sector_year_mean'] = df.groupby(['القطاع', 'السنة'])['المبيعات_بعد_التضخم'].transform('mean')
df['sector_year_std'] = df.groupby(['القطاع', 'السنة'])['المبيعات_بعد_التضخم'].transform('std')
df['z_within_industry_year'] = (df['المبيعات_بعد_التضخم'] - df['sector_year_mean']) / df['sector_year_std']
df['z_within_industry_year'] = df['z_within_industry_year'].fillna(0)

#percentiles by sector-year for each metric
metrics_percentiles = {
    'مبيعات_لكل_موظف': [10, 20, 97.5, 99, 99.5],
    'الموظفون': [10, 20, 90, 95],
    'رأس_المال_المدفوع_جنيه': [90, 95],
    'العائد_على_رأس_المال': [5, 10, 15, 25]
}

for metric, percentiles in metrics_percentiles.items():
    for p in percentiles:
        col_name = f'{metric}_p{p}'
        df[col_name] = df.groupby(['القطاع', 'السنة'])[metric].transform('quantile', p/100)

df['R1'] = 0
df['R2'] = 0
df['R3'] = 0
df['R4'] = 0
df['S1'] = 0
df['S2'] = 0
df['S3'] = 0
df['S4'] = 0

for idx in df.index:
    sector = df.at[idx, 'القطاع']
    
    #R1:Sales Spike Detection
    if sector == SECTOR_ICT:
        r1_threshold, r1_z = 6.0, 3.5
    elif sector == SECTOR_AGRICULTURE:
        r1_threshold, r1_z = 2.5, 2.5
    elif sector == SECTOR_MINING:
        r1_threshold, r1_z = 5.0, 3.0
    else:
        r1_threshold, r1_z = 4.0, 3.0
    
    if (df.at[idx, 'النمو_الحقيقي_للمبيعات'] >= r1_threshold and 
        df.at[idx, 'z_within_industry_year'] >= r1_z and 
        df.at[idx, 'نمو_الموظفين'] <= 0.2):
        df.at[idx, 'R1'] = 1
    
    #R2:High Sales + Low Employees
    r2_col = 'مبيعات_لكل_موظف_p99.5' if sector == SECTOR_ICT else 'مبيعات_لكل_موظف_p99'
    if (df.at[idx, 'مبيعات_لكل_موظف'] >= df.at[idx, r2_col] and 
        df.at[idx, 'الموظفون'] <= df.at[idx, 'الموظفون_p10'] and 
        df.at[idx, 'z_within_industry_year'] >= 2.5):
        df.at[idx, 'R2'] = 1
    
    #R3:High Employees + Low Sales
    if (df.at[idx, 'الموظفون'] >= df.at[idx, 'الموظفون_p95'] and 
        df.at[idx, 'مبيعات_لكل_موظف'] <= df.at[idx, 'مبيعات_لكل_موظف_p10'] and 
        df.at[idx, 'نمو_المبيعات'] <= 0.1):
        df.at[idx, 'R3'] = 1
    
    #R4:Large capital with poor utilization
    r4_col = 'العائد_على_رأس_المال_p5' if sector == SECTOR_MINING else 'العائد_على_رأس_المال_p10'
    if (df.at[idx, 'رأس_المال_المدفوع_جنيه'] >= df.at[idx, 'رأس_المال_المدفوع_جنيه_p95'] and 
        df.at[idx, 'العائد_على_رأس_المال'] <= df.at[idx, r4_col] and 
        abs(df.at[idx, 'النمو_الحقيقي_للمبيعات']) <= 0.05 and 
        df.at[idx, 'العائد_على_رأس_المال'] <= df.at[idx, 'العائد_على_رأس_المال_p10']):
        df.at[idx, 'R4'] = 1
    
    #S1:Sales Spike Detection
    if sector == SECTOR_ICT:
        s1_threshold, s1_z = 3.5, 2.5
    elif sector == SECTOR_AGRICULTURE:
        s1_threshold, s1_z = 1.5, 1.5
    elif sector == SECTOR_MINING:
        s1_threshold, s1_z = 3.0, 2.0
    else:
        s1_threshold, s1_z = 2.0, 2.0
    
    if (df.at[idx, 'النمو_الحقيقي_للمبيعات'] >= s1_threshold and 
        df.at[idx, 'z_within_industry_year'] >= s1_z and 
        df.at[idx, 'نمو_الموظفين'] <= 0.3):
        df.at[idx, 'S1'] = 1
    
    #S2:High Sales + Low Employees
    s2_col = 'مبيعات_لكل_موظف_p99' if sector == SECTOR_ICT else 'مبيعات_لكل_موظف_p97.5'
    if (df.at[idx, 'مبيعات_لكل_موظف'] >= df.at[idx, s2_col] and 
        df.at[idx, 'الموظفون'] <= df.at[idx, 'الموظفون_p20'] and 
        df.at[idx, 'z_within_industry_year'] >= 2.0):
        df.at[idx, 'S2'] = 1
    
    #S3:High Employees + Low Sales
    if (df.at[idx, 'الموظفون'] >= df.at[idx, 'الموظفون_p90'] and 
        df.at[idx, 'مبيعات_لكل_موظف'] <= df.at[idx, 'مبيعات_لكل_موظف_p20']):
        df.at[idx, 'S3'] = 1
    
    #S4:High Capital + Stagnant Sales
    s4_roc_col = 'العائد_على_رأس_المال_p15' if sector == SECTOR_MINING else 'العائد_على_رأس_المال_p25'
    if (df.at[idx, 'رأس_المال_المدفوع_جنيه'] >= df.at[idx, 'رأس_المال_المدفوع_جنيه_p90'] and 
        df.at[idx, 'العائد_على_رأس_المال'] <= df.at[idx, 'العائد_على_رأس_المال_p15'] and 
        abs(df.at[idx, 'النمو_الحقيقي_للمبيعات']) <= 0.10 and 
        df.at[idx, 'العائد_على_رأس_المال'] <= df.at[idx, s4_roc_col]):
        df.at[idx, 'S4'] = 1


#output
df['conservative_anomaly'] = (df[['R1', 'R2', 'R3', 'R4']].sum(axis=1) > 0).astype(int)
df['sensitive_anomaly'] = (df[['S1', 'S2', 'S3', 'S4']].sum(axis=1) > 0).astype(int)
df['عدد_القواعد_R'] = df[['R1', 'R2', 'R3', 'R4']].sum(axis=1)
df['عدد_القواعد_S'] = df[['S1', 'S2', 'S3', 'S4']].sum(axis=1)

original_cols = ['الرقم_الضريبي', 'السنة', 'القطاع', 'القسم_كود', 'المجموع_كود', 'الفرع_كود',
                 'المبيعات_جنيه', 'المبيعات_بعد_التضخم', 'الإيرادات_جنيه', 'الموظفون',
                 'رأس_المال_المدفوع_جنيه', 'ضريبة_القيمة_المضافة_المتوقعة',
                 'ضريبة_القيمة_المضافة_المصرح_بها', 'عدد الفروع', 'فئة_SME', 'سنة_البداية',
                 'اسم_الشركة', 'عمر_المنشأة', 'نمو_المبيعات', 'النمو_الحقيقي_للمبيعات',
                 'نمو_الموظفين', 'العائد_على_رأس_المال', 'score', 'نسبة_التضخم', 'مبيعات_لكل_موظف']

rule_cols = ['conservative_anomaly', 'sensitive_anomaly', 'R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4', 'عدد_القواعد_R', 'عدد_القواعد_S']
df = df[original_cols + rule_cols]

df.to_csv("data/anomaly_detection_results.csv", index=False, encoding='utf-8-sig')

print("Anomaly detection completed!")
print(f"Total rows: {len(df)}")
print(f"Rows with conservative anomalies: {df['conservative_anomaly'].sum()}")
print(f"Rows with sensitive anomalies: {df['sensitive_anomaly'].sum()}")
print(f"\nConservative rules breakdown:")
print(f"R1: {df['R1'].sum()}")
print(f"R2: {df['R2'].sum()}")
print(f"R3: {df['R3'].sum()}")
print(f"R4: {df['R4'].sum()}")
print(f"\nSensitive rules breakdown:")
print(f"S1: {df['S1'].sum()}")
print(f"S2: {df['S2'].sum()}")
print(f"S3: {df['S3'].sum()}")
print(f"S4: {df['S4'].sum()}")