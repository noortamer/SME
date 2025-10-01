import pandas as pd

df = pd.read_csv("data/finalscores.csv")

df.sort_values(by=["الرقم_الضريبي", "السنة"], inplace=True)

inflation_rates = {
    2025: 0.2395,
    2024: 0.3571,
    2023: 0.31932,
    2022: 0.07259,
    2021: 0.04285
}

df["نسبة_التضخم"] = df["السنة"].map(inflation_rates)

first_year_mask = ~df.duplicated(subset=["الرقم_الضريبي"], keep="first")
df.loc[first_year_mask, "نسبة_التضخم"] = 0

df["المبيعات_بعد_التضخم"] = df["المبيعات_جنيه"] / (1 + df["نسبة_التضخم"])

df["النمو_الحقيقي_للمبيعات"] = df.groupby("الرقم_الضريبي")["المبيعات_بعد_التضخم"].pct_change()



# Organize the csv better
df.drop(columns=["المبيعات_جنيه_p", "نمو_المبيعات_p", "الموظفون_p", "نمو_الموظفين_p", 
                 "عمر_المنشأة_p", "العائد_على_رأس_المال_p", "branches_p", 
                 "القسم", "المجموع", "الفرع"], inplace=True, errors='ignore')

df.rename(columns={"branches": "عدد الفروع", "start_year": "سنة_البداية"}, inplace=True)

cols = df.columns.tolist()
sales_idx = cols.index("المبيعات_جنيه")
cols.insert(sales_idx + 1, cols.pop(cols.index("المبيعات_بعد_التضخم")))

growth_idx = cols.index("نمو_المبيعات")
cols.insert(growth_idx + 1, cols.pop(cols.index("النمو_الحقيقي_للمبيعات")))

df = df[cols]

df.to_csv("data/scores_after_inflation.csv", index=False, encoding="utf-8-sig")