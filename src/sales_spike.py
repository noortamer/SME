import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('data/finalscores.csv')
company_benchmarks = pd.read_csv('data/company_benchmarks.csv')
industry_trends = pd.read_csv('data/industry_trends.csv')
industry_benchmarks = pd.read_csv('data/industry_benchmarks.csv')

sales_col = 'المبيعات_جنيه'
year_col = 'السنة'
sector_col = 'القطاع'
sme_col = 'فئة_SME'
tax_id_col = 'الرقم_الضريبي'
growth_col = 'نمو_المبيعات'

df['sales_yoy_growth'] = df[growth_col]

df = df.merge(industry_trends[[sector_col, year_col, 'industry_yoy_change']], on=[sector_col, year_col], how='left')
df = df.rename(columns={'industry_yoy_change': 'industry_avg_yoy_growth'})

df = df.merge(company_benchmarks[[tax_id_col, year_col, 'z_within_industry_year', sme_col]], on=[tax_id_col, year_col, sme_col], how='left')

df['spike_anomaly'] = np.where((df['sales_yoy_growth'] > 5.0) & (df['z_within_industry_year'] > 2), 1, 0)

industry_benchmarks['overall_mean'] = industry_benchmarks['year_industry_mean'].mean()
industry_benchmarks['industry_z'] = stats.zscore(industry_benchmarks['year_industry_mean'])
industry_benchmarks['industry_spike_anomaly'] = np.where(np.abs(industry_benchmarks['industry_z']) > 2, 1, 0)

df = df.merge(industry_benchmarks[[sector_col, year_col, 'industry_spike_anomaly']], on=[sector_col, year_col], how='left')

df.to_csv('data/finalscores_with_spike_anomalies.csv', index=False, encoding="utf-8-sig")
print("Spike anomalies detected and saved to 'finalscores_with_spike_anomalies.csv'")