import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates=True)
print("head : \n", bitcoin.head())
print("shape : \n", bitcoin.shape)

bitcoin['Close'].plot(figsize=(9, 6))
plt.show()

print("index : \n ", bitcoin.index)

bitcoin.loc['2019', 'Close'].resample('M').plot()
plt.show()


#########################################################
#  moving averted
plt.figure(figsize=(12, 8))
bitcoin.loc['2019', 'Close'].plot()
# alpha : lissage , t=t
bitcoin.loc['2019', 'Close'].resample('M').mean().plot(
    label='moyenne par mois', lw=3, ls=':', alpha=0.8)
bitcoin.loc['2019', 'Close'].resample('W').mean().plot(
    label='moyenne par semaine', lw=2, ls='--', alpha=0.8)
plt.legend()
plt.show()

# ETH
ethereum = pd.read_csv('ETH-EUR.csv', index_col='Date', parse_dates=True)

btc_eth = pd.merge(bitcoin, ethereum, on='Date',
                   how='inner', suffixes=('_btc', '_eth'))
btc_eth[['Close_btc', 'Close_eth']
        ]['2019'].plot(subplots=True, figsize=(12, 8))
plt.show()
print("corrr : \n ", btc_eth[['Close_btc', 'Close_eth']
                             ]['2019'] .corr())
