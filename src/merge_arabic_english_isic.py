import pandas as pd

df_en = pd.read_csv('data/companies_enriched.csv')
df_ar = pd.read_csv('data/الشركات_ar_enriched_fixed.csv')

section_en = df_en[['isic_class', 'sector']].drop_duplicates()
section_en['sector_code'] = section_en['isic_class'].str[0]
section_en = section_en[['sector_code', 'sector']].drop_duplicates().rename(columns={'sector': 'sector_en'})

section_ar = df_ar[['الفرع_كود', 'القطاع']].drop_duplicates()
section_ar['sector_code'] = section_ar['الفرع_كود'].str[0]
section_ar = section_ar[['sector_code', 'القطاع']].drop_duplicates().rename(columns={'القطاع': 'sector_ar'})

sections = pd.merge(section_en, section_ar, on='sector_code', how='outer').drop_duplicates()

division_ar = df_ar[['القسم_كود', 'القسم']].drop_duplicates().rename(columns={'القسم_كود': 'division_code', 'القسم': 'division_ar'})

group_ar = df_ar[['المجموع_كود', 'المجموع']].drop_duplicates().rename(columns={'المجموع_كود': 'group_code', 'المجموع': 'group_ar'})

class_en = df_en[['isic_class', 'isic_class_name_en']].drop_duplicates().rename(columns={'isic_class': 'class_code', 'isic_class_name_en': 'class_en'})

class_ar = df_ar[['الفرع_كود', 'الفرع']].drop_duplicates().rename(columns={'الفرع_كود': 'class_code', 'الفرع': 'class_ar'})

classes = pd.merge(class_en, class_ar, on='class_code', how='outer').sort_values('class_code')

classes['sector_code'] = classes['class_code'].str[0]
classes['division_code'] = classes['class_code'].str[0:2]
classes['group_code'] = classes['class_code'].str[0:3]

classes = classes.merge(sections, on='sector_code', how='left')
classes = classes.merge(division_ar, on='division_code', how='left')
classes = classes.merge(group_ar, on='group_code', how='left')

result = classes[['sector_en', 'sector_ar', 'sector_code', 'division_ar', 'division_code', 'group_ar', 'group_code', 'class_en', 'class_ar', 'class_code']].drop_duplicates().sort_values('class_code')

result.to_csv('data/isic_ar_en.csv', index=False, encoding='utf-8-sig')

print("ISIC hierarchy mapping completed!")
print(f"Total class-level records: {len(result)}")