import pandas as pd
import numpy as np

df = pd.read_csv("data/finalscores.csv")

# List of columns to process
columns = [
    'المبيعات_جنيه', 'الإيرادات_جنيه', 'الموظفون', 'رأس_المال_المدفوع_جنيه',
    'ضريبة_القيمة_المضافة_المتوقعة', 'ضريبة_القيمة_المضافة_المصرح_بها',
    'branches', 'عمر_المنشأة', 'نمو_المبيعات', 'نمو_الموظفين', 'العائد_على_رأس_المال',
    'المبيعات_جنيه_p', 'نمو_المبيعات_p', 'الموظفون_p', 'نمو_الموظفين_p',
    'عمر_المنشأة_p', 'العائد_على_رأس_المال_p', 'branches_p', 'score'
]

# Create a unique identifier column
df['unique_id'] = df['الرقم_الضريبي'].astype(str) + '-' + df['السنة'].astype(str)

# Prepare a list to hold results
results = []

for col in columns:
    series = pd.to_numeric(df[col], errors='coerce')
    min_value = series.min()
    max_value = series.max()
    mean_value = series.mean()
    median_value = series.median()
    p75 = series.quantile(0.75)
    p95 = series.quantile(0.95)
    p99 = series.quantile(0.99)

    # Get indexes of min and max
    min_idx = series.idxmin()
    max_idx = series.idxmax()
    if pd.notnull(min_idx):
        min_id = df.loc[min_idx, 'unique_id']
    else:
        min_id = np.nan
    if pd.notnull(max_idx):
        max_id = df.loc[max_idx, 'unique_id']
    else:
        max_id = np.nan

    results.append({
        'column': col,
        'min': min_value,
        'min_id': min_id,
        'max': max_value,
        'max_id': max_id,
        'mean': mean_value,
        'median': median_value,
        '75th_percentile': p75,
        '95th_percentile': p95,
        '99th_percentile': p99
    })

# Create result DataFrame
stats_df = pd.DataFrame(results)

# Export to CSV
stats_df.to_csv('analysis/company_stats_summary.csv', index=False, encoding='utf-8-sig')
