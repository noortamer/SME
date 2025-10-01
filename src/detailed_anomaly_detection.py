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

# Initialize anomaly flags
df['R1'] = 0
df['R2'] = 0
df['R3'] = 0
df['R4'] = 0
df['S1'] = 0
df['S2'] = 0
df['S3'] = 0
df['S4'] = 0

# Initialize equation columns
equation_cols = ['R1_equation', 'R2_equation', 'R3_equation', 'R4_equation', 
                'S1_equation', 'S2_equation', 'S3_equation', 'S4_equation']
for col in equation_cols:
    df[col] = ''

def format_equation(condition, result, threshold=None, comparison_value=None):
    """Format equation as a mathematical comparison"""
    if threshold is not None:
        if comparison_value is not None:
            return f"{condition} = {result} {'≥' if result >= threshold else '<'} {threshold}"
        return f"{condition} = {result} {'≥' if result >= threshold else '<'} {threshold}"
    elif comparison_value is not None:
        return f"{condition} = {result} {'≥' if result >= comparison_value else '<'} {comparison_value}"
    return f"{condition} = {result}"

def format_multi_condition_equation(conditions, all_true):
    """Format multiple conditions for AND logic"""
    formatted = []
    for i, (cond, val, thresh, comp_val) in enumerate(conditions):
        if thresh is not None:
            comparison = '≥' if val >= thresh else '<'
            formatted.append(f"C{i+1}: {cond} = {val} {comparison} {thresh}")
        elif comp_val is not None:
            comparison = '≥' if val >= comp_val else '<'
            formatted.append(f"C{i+1}: {cond} = {val} {comparison} {comp_val}")
        else:
            formatted.append(f"C{i+1}: {cond} = {val}")
    
    result_str = "TRUE" if all_true else "FALSE"
    return f"({' & '.join(formatted)}) = {result_str}"

# Process each row
for idx in df.index:
    sector = df.at[idx, 'القطاع']
    tax_id = df.at[idx, 'الرقم_الضريبي']
    year = df.at[idx, 'السنة']
    
    # Extract values for this row
    sales_growth = df.at[idx, 'النمو_الحقيقي_للمبيعات']
    z_score = df.at[idx, 'z_within_industry_year']
    employee_growth = df.at[idx, 'نمو_الموظفين']
    sales_per_employee = df.at[idx, 'مبيعات_لكل_موظف']
    employees = df.at[idx, 'الموظفون']
    capital = df.at[idx, 'رأس_المال_المدفوع_جنيه']
    roc = df.at[idx, 'العائد_على_رأس_المال']
    nominal_sales_growth = df.at[idx, 'نمو_المبيعات']
    
    # R1: Sales Spike Detection
    if sector == SECTOR_ICT:
        r1_threshold, r1_z = 6.0, 3.5
    elif sector == SECTOR_AGRICULTURE:
        r1_threshold, r1_z = 2.5, 2.5
    elif sector == SECTOR_MINING:
        r1_threshold, r1_z = 5.0, 3.0
    else:
        r1_threshold, r1_z = 4.0, 3.0
    
    r1_conditions = [
        ("النمو_الحقيقي_للمبيعات", sales_growth, r1_threshold, None),
        ("z_within_industry_year", z_score, r1_z, None),
        ("نمو_الموظفين ≤ 0.2", employee_growth, 0.2, None)
    ]
    
    r1_true = (
        sales_growth >= r1_threshold and 
        z_score >= r1_z and 
        employee_growth <= 0.2
    )
    
    if r1_true:
        df.at[idx, 'R1'] = 1
    df.at[idx, 'R1_equation'] = format_multi_condition_equation(r1_conditions, r1_true)
    
    # R2: High Sales + Low Employees
    r2_col = 'مبيعات_لكل_موظف_p99.5' if sector == SECTOR_ICT else 'مبيعات_لكل_موظف_p99'
    r2_sales_threshold = df.at[idx, r2_col]
    r2_employee_threshold = df.at[idx, 'الموظفون_p10']
    r2_z_threshold = 2.5
    
    r2_conditions = [
        ("مبيعات_لكل_موظف", sales_per_employee, None, r2_sales_threshold),
        ("الموظفون", employees, None, r2_employee_threshold),
        ("z_within_industry_year", z_score, r2_z_threshold, None)
    ]
    
    r2_true = (
        sales_per_employee >= r2_sales_threshold and 
        employees <= r2_employee_threshold and 
        z_score >= r2_z_threshold
    )
    
    if r2_true:
        df.at[idx, 'R2'] = 1
    df.at[idx, 'R2_equation'] = format_multi_condition_equation(r2_conditions, r2_true)
    
    # R3: High Employees + Low Sales
    r3_employee_threshold = df.at[idx, 'الموظفون_p95']
    r3_sales_per_emp_threshold = df.at[idx, 'مبيعات_لكل_موظف_p10']
    r3_nominal_growth_threshold = 0.1
    
    r3_conditions = [
        ("الموظفون", employees, None, r3_employee_threshold),
        ("مبيعات_لكل_موظف", sales_per_employee, None, r3_sales_per_emp_threshold),
        ("نمو_المبيعات", nominal_sales_growth, r3_nominal_growth_threshold, None)
    ]
    
    r3_true = (
        employees >= r3_employee_threshold and 
        sales_per_employee <= r3_sales_per_emp_threshold and 
        nominal_sales_growth <= r3_nominal_growth_threshold
    )
    
    if r3_true:
        df.at[idx, 'R3'] = 1
    df.at[idx, 'R3_equation'] = format_multi_condition_equation(r3_conditions, r3_true)
    
    # R4: Large capital with poor utilization
    r4_col = 'العائد_على_رأس_المال_p5' if sector == SECTOR_MINING else 'العائد_على_رأس_المال_p10'
    r4_roc_threshold = df.at[idx, r4_col]
    r4_capital_threshold = df.at[idx, 'رأس_المال_المدفوع_جنيه_p95']
    r4_sales_growth_threshold = 0.05
    r4_general_roc_threshold = df.at[idx, 'العائد_على_رأس_المال_p10']
    
    r4_conditions = [
        ("رأس_المال_المدفوع_جنيه", capital, None, r4_capital_threshold),
        ("العائد_على_رأس_المال", roc, None, r4_roc_threshold),
        ("|النمو_الحقيقي_للمبيعات| ≤ 0.05", abs(sales_growth), r4_sales_growth_threshold, None),
        ("العائد_على_رأس_المال", roc, None, r4_general_roc_threshold)
    ]
    
    r4_true = (
        capital >= r4_capital_threshold and 
        roc <= r4_roc_threshold and 
        abs(sales_growth) <= r4_sales_growth_threshold and 
        roc <= r4_general_roc_threshold
    )
    
    if r4_true:
        df.at[idx, 'R4'] = 1
    df.at[idx, 'R4_equation'] = format_multi_condition_equation(r4_conditions, r4_true)
    
    # S1: Sales Spike Detection (Sensitive)
    if sector == SECTOR_ICT:
        s1_threshold, s1_z = 3.5, 2.5
    elif sector == SECTOR_AGRICULTURE:
        s1_threshold, s1_z = 1.5, 1.5
    elif sector == SECTOR_MINING:
        s1_threshold, s1_z = 3.0, 2.0
    else:
        s1_threshold, s1_z = 2.0, 2.0
    
    s1_conditions = [
        ("النمو_الحقيقي_للمبيعات", sales_growth, s1_threshold, None),
        ("z_within_industry_year", z_score, s1_z, None),
        ("نمو_الموظفين ≤ 0.3", employee_growth, 0.3, None)
    ]
    
    s1_true = (
        sales_growth >= s1_threshold and 
        z_score >= s1_z and 
        employee_growth <= 0.3
    )
    
    if s1_true:
        df.at[idx, 'S1'] = 1
    df.at[idx, 'S1_equation'] = format_multi_condition_equation(s1_conditions, s1_true)
    
    # S2: High Sales + Low Employees (Sensitive)
    s2_col = 'مبيعات_لكل_موظف_p99' if sector == SECTOR_ICT else 'مبيعات_لكل_موظف_p97.5'
    s2_sales_threshold = df.at[idx, s2_col]
    s2_employee_threshold = df.at[idx, 'الموظفون_p20']
    s2_z_threshold = 2.0
    
    s2_conditions = [
        ("مبيعات_لكل_موظف", sales_per_employee, None, s2_sales_threshold),
        ("الموظفون", employees, None, s2_employee_threshold),
        ("z_within_industry_year", z_score, s2_z_threshold, None)
    ]
    
    s2_true = (
        sales_per_employee >= s2_sales_threshold and 
        employees <= s2_employee_threshold and 
        z_score >= s2_z_threshold
    )
    
    if s2_true:
        df.at[idx, 'S2'] = 1
    df.at[idx, 'S2_equation'] = format_multi_condition_equation(s2_conditions, s2_true)
    
    # S3: High Employees + Low Sales (Sensitive)
    s3_employee_threshold = df.at[idx, 'الموظفون_p90']
    s3_sales_per_emp_threshold = df.at[idx, 'مبيعات_لكل_موظف_p20']
    
    s3_conditions = [
        ("الموظفون", employees, None, s3_employee_threshold),
        ("مبيعات_لكل_موظف", sales_per_employee, None, s3_sales_per_emp_threshold)
    ]
    
    s3_true = (
        employees >= s3_employee_threshold and 
        sales_per_employee <= s3_sales_per_emp_threshold
    )
    
    if s3_true:
        df.at[idx, 'S3'] = 1
    df.at[idx, 'S3_equation'] = format_multi_condition_equation(s3_conditions, s3_true)
    
    # S4: High Capital + Stagnant Sales (Sensitive)
    s4_roc_col = 'العائد_على_رأس_المال_p15' if sector == SECTOR_MINING else 'العائد_على_رأس_المال_p25'
    s4_roc_threshold = df.at[idx, s4_roc_col]
    s4_capital_threshold = df.at[idx, 'رأس_المال_المدفوع_جنيه_p90']
    s4_sales_growth_threshold = 0.10
    s4_general_roc_threshold = df.at[idx, 'العائد_على_رأس_المال_p15']
    
    s4_conditions = [
        ("رأس_المال_المدفوع_جنيه", capital, None, s4_capital_threshold),
        ("العائد_على_رأس_المال", roc, None, s4_general_roc_threshold),
        ("|النمو_الحقيقي_للمبيعات| ≤ 0.10", abs(sales_growth), s4_sales_growth_threshold, None),
        ("العائد_على_رأس_المال", roc, None, s4_roc_threshold)
    ]
    
    s4_true = (
        capital >= s4_capital_threshold and 
        roc <= s4_general_roc_threshold and 
        abs(sales_growth) <= s4_sales_growth_threshold and 
        roc <= s4_roc_threshold
    )
    
    if s4_true:
        df.at[idx, 'S4'] = 1
    df.at[idx, 'S4_equation'] = format_multi_condition_equation(s4_conditions, s4_true)

# Calculate summary anomaly flags
df['conservative_anomaly'] = (df[['R1', 'R2', 'R3', 'R4']].sum(axis=1) > 0).astype(int)
df['sensitive_anomaly'] = (df[['S1', 'S2', 'S3', 'S4']].sum(axis=1) > 0).astype(int)
df['عدد_القواعد_R'] = df[['R1', 'R2', 'R3', 'R4']].sum(axis=1)
df['عدد_القواعد_S'] = df[['S1', 'S2', 'S3', 'S4']].sum(axis=1)

# Select final columns
original_cols = ['الرقم_الضريبي', 'السنة', 'القطاع', 'القسم_كود', 'المجموع_كود', 'الفرع_كود',
                 'المبيعات_جنيه', 'المبيعات_بعد_التضخم', 'الإيرادات_جنيه', 'الموظفون',
                 'رأس_المال_المدفوع_جنيه', 'ضريبة_القيمة_المضافة_المتوقعة',
                 'ضريبة_القيمة_المضافة_المصرح_بها', 'عدد الفروع', 'فئة_SME', 'سنة_البداية',
                 'اسم_الشركة', 'عمر_المنشأة', 'نمو_المبيعات', 'النمو_الحقيقي_للمبيعات',
                 'نمو_الموظفين', 'العائد_على_رأس_المال', 'score', 'نسبة_التضخم', 'مبيعات_لكل_موظف']

rule_cols = ['conservative_anomaly', 'sensitive_anomaly', 'R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4', 
             'عدد_القواعد_R', 'عدد_القواعد_S'] + equation_cols

df_final = df[original_cols + rule_cols]

# Save results
df_final.to_csv("data/anomaly_detection_results_detailed.csv", index=False, encoding='utf-8-sig')

# Print summary
print("Anomaly detection completed with detailed equations!")
print(f"Total rows: {len(df_final)}")
print(f"Rows with conservative anomalies: {df_final['conservative_anomaly'].sum()}")
print(f"Rows with sensitive anomalies: {df_final['sensitive_anomaly'].sum()}")

print(f"\nConservative rules breakdown:")
print(f"R1: {df_final['R1'].sum()}")
print(f"R2: {df_final['R2'].sum()}")
print(f"R3: {df_final['R3'].sum()}")
print(f"R4: {df_final['R4'].sum()}")

print(f"\nSensitive rules breakdown:")
print(f"S1: {df_final['S1'].sum()}")
print(f"S2: {df_final['S2'].sum()}")
print(f"S3: {df_final['S3'].sum()}")
print(f"S4: {df_final['S4'].sum()}")

# Show sample equations for anomalies
print(f"\nSample equations for anomalies (first 5 rows with any anomaly):")
anomaly_rows = df_final[df_final[['R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4']].sum(axis=1) > 0].head(5)

for idx in anomaly_rows.index:
    tax_id = df_final.at[idx, 'الرقم_الضريبي']
    sector = df_final.at[idx, 'القطاع']
    year = df_final.at[idx, 'السنة']
    
    print(f"\n--- Row: Tax ID {tax_id}, Sector: {sector}, Year: {year} ---")
    active_rules = []
    for rule in ['R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4']:
        if df_final.at[idx, rule] == 1:
            equation = df_final.at[idx, f'{rule}_equation']
            active_rules.append(f"{rule}: {equation}")
    
    for rule_eq in active_rules:
        print(f"  {rule_eq}")

print(f"\nDetailed results saved to: data/anomaly_detection_results_detailed.csv")
print("Each row now includes equation columns showing exact calculations and comparisons for each rule.")