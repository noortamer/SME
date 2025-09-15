import pandas as pd

file_path = r"data\scores_peers.csv"
df = pd.read_csv(file_path)

df = df.sort_values(by=["الرقم_الضريبي", "السنة"])

def calculate_historical_sales_score(group):
    #calculate percentage change
    group = group.copy()
    group["historical_sales_score"] = group["المبيعات_جنيه"].pct_change() * 100
    
    group.loc[group.index[0], "historical_sales_score"] = 0
    
    #handle NaN values
    group["historical_sales_score"] = group["historical_sales_score"].fillna(0)
    group["historical_sales_score"] = group["historical_sales_score"].replace([float('inf'), float('-inf')], 0)
    
    #round to 2 decimal places
    group["historical_sales_score"] = group["historical_sales_score"].round(2)
    
    return group

#apply the calculation for each company (grouped by "الرقم_الضريبي")
df = df.groupby("الرقم_الضريبي", group_keys=False).apply(calculate_historical_sales_score)
df = df.reset_index(drop=True)

output_path = r"data\historical_sales_scores.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Historical sales scores calculated and saved to {output_path}")
print(f"\nDataset shape: {df.shape}")
print(f"Companies analyzed: {df['الرقم_الضريبي'].nunique()}")
print(f"Years covered: {df['السنة'].min()} - {df['السنة'].max()}")
print("\nHistorical Sales Score statistics:")
print(df['historical_sales_score'].describe())

print(f"\nSample of companies with their historical sales scores:")
sample_companies = df['الرقم_الضريبي'].unique()[:3]
for company in sample_companies:
    company_data = df[df['الرقم_الضريبي'] == company][['السنة', 'المبيعات_جنيه', 'historical_sales_score']]
    print(f"\nCompany {company}:")
    print(company_data.to_string(index=False))