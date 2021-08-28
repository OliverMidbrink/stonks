import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math

# Show stock price adjusted for dividend

"""
GSPC = yf.Ticker("^GSPC")
GSPC_hist = GSPC.history(period="max")
GSPC_hist['avg_high_low'] = GSPC_hist[["High", "Low"]].mean(axis=1)
GSPC_hist['ma_high_low'] = GSPC_hist['avg_high_low'].rolling(window=253).mean()

DJI = yf.Ticker("^DJI")
DJI_hist = DJI.history(start="1992-03-01")
DJI_hist['avg_high_low'] = DJI_hist[["High", "Low"]].mean(axis=1)
DJI_hist['ma_high_low'] = DJI_hist['avg_high_low'].rolling(window=253).mean()

OMXS30 = yf.Ticker("^OMXS30")
OMXS30_hist = OMXS30.history(start="1980-01-01", end="2100-01-01")
OMXS30_hist['avg_high_low'] = OMXS30_hist[["High", "Low"]].mean(axis=1)
OMXS30_hist['ma_high_low'] = OMXS30_hist['avg_high_low'].rolling(window=253).mean()

print(OMXS30_hist['avg_high_low'])
"""

def dividend_adjusted_history(ticker_name):
    ticker = yf.Ticker(ticker_name)

    dividends = ticker.dividends
    

    hist = ticker.history(period="5y", auto_adjust=False)
    hist['avg_high_low'] = hist[["High", "Low"]].mean(axis=1)

    pct_change = hist["avg_high_low"].pct_change()
    hist['adj_price'] = hist['avg_high_low']
    
    start_time = time.time()

    # Iterate all available stock market days, start at 1 since 0 will be used as index. Index start is 1
    for date_idx in range(1, hist.shape[0], 1):
        new_adj_price = (pct_change.iloc[date_idx] + 1) * (hist.iloc[date_idx - 1]['adj_price'] + hist.iloc[date_idx - 1]['Dividends'])
        hist['adj_price'][date_idx] = new_adj_price

    print(hist)
    print(ticker.splits)
    print(ticker.dividends)
    price_development = hist.iloc[-1]['avg_high_low']/hist.iloc[0]['avg_high_low']
    print(price_development)
    print(hist.iloc[-1]['adj_price']/price_development)
    print("Calculation took: " + str(time.time() - start_time) + " seconds")

    return hist


start = "2014-06-01"
GSPC = yf.Ticker("^GSPC").history(start=start, end="2100-01-01")["Close"]
auto_adjustA = yf.Ticker("AAPL").history(start=start)["Close"]
auto_adjustB = yf.Ticker("ACN").history(start=start)["Close"]
print(GSPC)


# Normalize so that both start at 1
auto_adjustA /= float(auto_adjustA[:1])
auto_adjustB /= float(auto_adjustB[:1])
GSPC /= float(GSPC[:1])

print(auto_adjustA.std())
print(auto_adjustB.std())

# regression with exponential curve
def exp_reg_curve(input_series):
    reg_vals = np.polyfit(range(len(input_series.index)), np.log(input_series), 1, w=np.sqrt(input_series))
    
    def exp_func(x):
        y = np.exp(reg_vals[1]) * np.exp(reg_vals[0] * x)
        return y

    return pd.Series(data=exp_func(np.array(range(len(input_series.index)))), index=input_series.index)

reg_A = exp_reg_curve(auto_adjustA)
reg_B = exp_reg_curve(auto_adjustB)

print(reg_A)

fig, ax = plt.subplots()
ax.plot(auto_adjustA)
ax.plot(auto_adjustB)
ax.plot(reg_A)
ax.plot(reg_B)


ax.set(xlabel='Date', ylabel='Index',
       title='Stock Market Overview')
ax.grid()

plt.show()
