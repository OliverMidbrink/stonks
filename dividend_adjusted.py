import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Show stock price adjusted for dividend

GSPC = yf.Ticker("^GSPC")
GSPC_hist = GSPC.history(period="max")
GSPC_hist['avg_high_low'] = GSPC_hist[["High", "Low"]].mean(axis=1)
GSPC_hist['ma_high_low'] = GSPC_hist['avg_high_low'].rolling(window=253).mean()

DJI = yf.Ticker("^DJI")
DJI_hist = DJI.history(start="1992-03-01")
DJI_hist['avg_high_low'] = DJI_hist[["High", "Low"]].mean(axis=1)
DJI_hist['ma_high_low'] = DJI_hist['avg_high_low'].rolling(window=253).mean()


def do_stuff(ticker_name):
    ticker = yf.Ticker(ticker_name)

    dividends = ticker.dividends
    

    hist = ticker.history(period="max")
    hist['avg_high_low'] = hist[["High", "Low"]].mean(axis=1)

    pct_change = hist["avg_high_low"].pct_change()
    

    # Iterate all available stock market days, start at 1 since 0 will be used as index. Index start is 1
    for date in hist.index[1:]:
        hist['adj_price'] = 1

    print(hist)

do_stuff("AAPL")

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