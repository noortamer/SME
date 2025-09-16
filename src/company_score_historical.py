import pandas as pd
import numpy as np

file_path = r"data\scores_peers.csv"
df = pd.read_csv(file_path)

df = df.sort_values(by=['الرقم_الضريبي', 'السنة']).reset_index(drop=True)

# current year score - last year score = current score
df['historical_score'] = df.groupby('الرقم_الضريبي')['score'].diff().round(2)

df['historical_score'].fillna(0, inplace=True)

output_path = r"data\historical_scores.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Historical scores calculated and saved to {output_path}")
print("\nHistorical Score Statistics:")
print(df['historical_score'].describe())