import pandas as pd
df = pd.read_csv('data/boruta/4.csv')
print('Confirmed: ', df['boruta_selected'].sum())
print('Tentative: ', df['boruta_tentative'].sum())
print('Total if union:', df['boruta_selected'].sum() + df['boruta_tentative'].sum())