import yfinance as yf
import pandas as pd
import requests
import numpy as np

def format_br_date(date):
    ano, mes, dia = date.split('-')
    return f"{dia}/{mes}/{ano}"

ALPHA_KEY = "D5TA3ORCI0CV6N6U" # NTAFIABWAZY10NXJ

# Ações
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]  

start_date = "2020-01-01"
end_date = "2023-01-01"

data_frames = []
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker
    data_frames.append(data)
historical_data = pd.concat(data_frames)
print(historical_data.head())

# Taxa Selic 
start_br = format_br_date(start_date)
end_br = format_br_date(end_date)

selic_url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={start_br}&dataFinal={end_br}"
response = requests.get(selic_url)
selic_data = response.json()

selic_df = pd.DataFrame(selic_data)
selic_df['data'] = pd.to_datetime(selic_df['data'], format='%d/%m/%Y')
selic_df.set_index('data', inplace=True)
selic_df.rename(columns={'valor': 'selic', 'data':'Date'}, inplace=True)
print(selic_df.head())

# Barril de Petróleo 
oil_url = f"https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={ALPHA_KEY}"
response = requests.get(oil_url)
oil_data = response.json()

oil_df = pd.DataFrame(oil_data['data'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
oil_df['value'] = oil_df['value'].replace('.', np.nan)
oil_df['value'] = oil_df['value'].astype(float)
oil_df.dropna(subset=['value'], inplace=True)
oil_df.rename(columns={'date': 'Date', 'value': 'oil_price'}, inplace=True)
oil_df.set_index('Date', inplace=True)
oil_df.sort_index(inplace=True)
oil_df = oil_df.loc[start_date:end_date]
print(oil_df.head())

# Câmbio USD/BRL
exchange_url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=BRL&outputsize=full&apikey={ALPHA_KEY}"
response = requests.get(exchange_url)
exchange_data = response.json()

exchange_rates = exchange_data['Time Series FX (Daily)']
exchange_df = pd.DataFrame.from_dict(exchange_rates, orient='index').astype(float)
exchange_df.rename(columns={'4. close': 'exchange_rate'}, inplace=True)
exchange_df.index = pd.to_datetime(exchange_df.index)
exchange_df.sort_index(inplace=True)
exchange_df = exchange_df[['exchange_rate']]
exchange_df = exchange_df.loc[start_date:end_date]
print(exchange_df.head())


# IPCA 

ipca_url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.10844/dados?formato=json&dataInicial={start_br}&dataFinal={end_br}"
response = requests.get(ipca_url)
ipca_data = response.json()

ipca_df = pd.DataFrame(ipca_data)
ipca_df['data'] = pd.to_datetime(ipca_df['data'], format='%d/%m/%Y')
ipca_df['valor'] = ipca_df['valor'].astype(float)
ipca_df.rename(columns={'data': 'Date', 'valor': 'ipca'}, inplace=True)

all_dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq='D')
all_dates_df = pd.DataFrame(all_dates, columns=['Date'])

ipca_df = all_dates_df.merge(ipca_df, on='Date', how='left')
ipca_df['ipca'].fillna(method='ffill', inplace=True)
ipca_df.set_index('Date', inplace=True)

# BVSP
bovespa_data = yf.download('^BVSP', start=start_date, end=end_date)
bovespa_df = bovespa_data[['Close']]
bovespa_df.rename(columns={'Close': 'bovespa'}, inplace=True)


# Combinando dfs
combined_data = historical_data.join(selic_df, on='Date', how='left')
combined_data = combined_data.join(oil_df, on='Date', how='left')
combined_data = combined_data.join(exchange_df, on='Date', how='left')
combined_data = combined_data.join(ipca_df, on='Date', how='left')
combined_data = combined_data.join(bovespa_df, on='Date', how='left')
combined_data.fillna(method='ffill', inplace=True)

combined_data.to_csv('combined_data.csv')