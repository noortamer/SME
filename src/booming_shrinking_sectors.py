import pandas as pd
import numpy as np

df = pd.read_csv('data/anomaly_detection_results.csv')
isic_mapping = pd.read_csv('data/isic_ar_en.csv')

df = df.merge(isic_mapping[['class_ar', 'class_code']], left_on='الفرع_كود', right_on='class_code', how='left')

df = df.sort_values(['class_ar', 'السنة'])

def calculate_sector_growth(group):
    sector_growth = group.groupby('السنة')['النمو_الحقيقي_للمبيعات'].mean()
    return sector_growth

sector_growth_dict = {}
for sector in df['class_ar'].unique():
    if pd.notna(sector):
        sector_data = df[df['class_ar'] == sector]
        sector_growth = calculate_sector_growth(sector_data)
        sector_growth_dict[sector] = sector_growth

def check_consecutive_years(sector, year, growth_dict, threshold, direction='boom'):
    if sector not in growth_dict or year not in growth_dict[sector].index:
        return 0
    
    consecutive = 0
    current_year = year
    
    current_growth = growth_dict[sector].get(current_year, np.nan)
    if pd.isna(current_growth):
        return 0
    
    if direction == 'boom' and current_growth <= threshold:
        return 0
    elif direction == 'shrink' and current_growth >= threshold:
        return 0
    
    while current_year in growth_dict[sector].index:
        growth_value = growth_dict[sector].get(current_year, np.nan)
        
        if pd.isna(growth_value):
            break
            
        if direction == 'boom' and growth_value > threshold:
            consecutive += 1
            current_year -= 1
        elif direction == 'shrink' and growth_value < threshold:
            consecutive += 1
            current_year -= 1
        else:
            break
    
    return consecutive

def check_shrinking_in_last_n_years(sector, year, growth_dict, threshold, n_years=5, min_shrinking_years=2):
    if sector not in growth_dict:
        return 0
    
    shrinking_count = 0
    
    for i in range(n_years):
        check_year = year - i
        if check_year in growth_dict[sector].index:
            growth_value = growth_dict[sector].get(check_year, np.nan)
            if pd.notna(growth_value) and growth_value < threshold:
                shrinking_count += 1
    
    return shrinking_count

latest_year = df['السنة'].max()
sector_summary = []

for sector in df['class_ar'].unique():
    if pd.isna(sector):
        continue
        
    sector_data = df[df['class_ar'] == sector]
    
    sector_years = sector_data['السنة'].unique()
    latest_sector_year = sector_years.max()
    
    boom_consecutive = check_consecutive_years(sector, latest_sector_year, sector_growth_dict, 0.10, 'boom')
    
    booming = 0
    booming_confidence = ''
    
    if boom_consecutive >= 4:
        booming = 1
        booming_confidence = 'very high'
    elif boom_consecutive >= 3:
        booming = 1
        booming_confidence = 'high'
    elif boom_consecutive >= 2:
        booming = 1
        booming_confidence = 'mid'
    
    shrink_count = check_shrinking_in_last_n_years(sector, latest_sector_year, sector_growth_dict, -0.01, 5, 2)
    
    shrinking = 0
    shrinking_confidence = ''
    
    if shrink_count >= 4:
        shrinking = 1
        shrinking_confidence = 'very high'
    elif shrink_count >= 3:
        shrinking = 1
        shrinking_confidence = 'high'
    elif shrink_count >= 2:
        shrinking = 1
        shrinking_confidence = 'mid'
    
    company_revenues = sector_data.groupby(['الرقم_الضريبي', 'اسم_الشركة'])['المبيعات_بعد_التضخم'].sum().reset_index()
    company_revenues = company_revenues.sort_values('المبيعات_بعد_التضخم', ascending=False)
    
    total_sector_revenue = company_revenues['المبيعات_بعد_التضخم'].sum()
    
    top_5_companies = company_revenues.head(5)
    
    top_companies_str = []
    for _, company in top_5_companies.iterrows():
        company_name = company['اسم_الشركة']
        revenue = company['المبيعات_بعد_التضخم']
        percentage = (revenue / total_sector_revenue * 100) if total_sector_revenue > 0 else 0
        top_companies_str.append(f"{company_name} ({percentage:.2f}%)")
    
    top_companies_formatted = " | ".join(top_companies_str)
    
    sector_summary.append({
        'class_ar': sector,
        'booming': booming,
        'booming_confidence': booming_confidence,
        'shrinking': shrinking,
        'shrinking_confidence': shrinking_confidence,
        'top_5_companies': top_companies_formatted
    })

sector_df = pd.DataFrame(sector_summary)

sector_df.to_csv('data/booming_shrinking_classes.csv', index=False, encoding='utf-8-sig')

print("Processing complete!")
print(f"Total unique classes: {len(sector_df)}")
print(f"\nBooming classes:")
print(sector_df[sector_df['booming'] == 1][['class_ar', 'booming_confidence']])
print(f"\nShrinking classes:")
print(sector_df[sector_df['shrinking'] == 1][['class_ar', 'shrinking_confidence']])
print(f"\nOutput saved to: data/class_boom_shrink_analysis.csv")