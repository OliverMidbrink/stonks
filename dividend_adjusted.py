import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Show stock price adjusted for dividend

GSPC = yf.Ticker("^GSPC")
GSPC_hist = GSPC.history(period="max")
GSPC_hist['avg_high_low'] = GSPC_hist[["High", "Low"]].mean(axis=1)
GSPC_hist['ma_high_low'] = GSPC_hist['avg_high_low'].rolling(window=253).mean()

DJI = yf.Ticker("^DJI")
DJI_hist = DJI.history(start="1992-03-01")
DJI_hist['avg_high_low'] = DJI_hist[["High", "Low"]].mean(axis=1)
DJI_hist['ma_high_low'] = DJI_hist['avg_high_low'].rolling(window=253).mean()


def dividend_adjusted_history(ticker_name):
    ticker = yf.Ticker(ticker_name)

    dividends = ticker.dividends
    

    hist = ticker.history(period="max")
    hist['avg_high_low'] = hist[["High", "Low"]].mean(axis=1)

    pct_change = hist["avg_high_low"].pct_change()
    hist['adj_price'] = 1.0
    
    start_time = time.time()

    # Iterate all available stock market days, start at 1 since 0 will be used as index. Index start is 1
    for date_idx in range(1, hist.shape[0], 1):
        new_adj_price = (pct_change.iloc[date_idx] + 1) * (hist.iloc[date_idx - 1]['adj_price'] + hist.iloc[date_idx - 1]['Dividends'] / hist.iloc[date_idx - 1]['avg_high_low'])
        hist['adj_price'][date_idx] = new_adj_price

    print(hist)
    print(ticker.splits)
    price_development = hist.iloc[-1]['avg_high_low']/hist.iloc[0]['avg_high_low']
    print(price_development)
    print(hist.iloc[-1]['adj_price']/price_development)
    print("Calculation took: " + str(time.time() - start_time) + " seconds")

dividend_adjusted_history("KAPIAB.ST")

"""
print((1 + GSPC_hist["avg_high_low"].pct_change().mean()) ** 253)
print((1 + DJI_hist["avg_high_low"].pct_change().mean()) ** 253)

fig, ax = plt.subplots()
ax.plot(GSPC_hist["ma_high_low"].pct_change())
ax.plot(DJI_hist["ma_high_low"].pct_change())



ax.set(xlabel='Date', ylabel='Index',
       title='Stock Market Overview')
ax.grid()

plt.show()
"""