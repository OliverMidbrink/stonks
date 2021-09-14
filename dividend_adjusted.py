from pandas.core.frame import DataFrame
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
def get_full_esg(ticker_name):
    return yesg.get_esg_full(ticker_name)


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

start = "1980-01-01"
A_ticker_name = "AAPL"
B_ticker_name = "CLAS-B.ST"
GSPC = yf.Ticker("^GSPC").history(start=start, end="2100-01-01")["Close"].fillna(method="ffill")

assetA_ticker = yf.Ticker(A_ticker_name)
assetB_ticker = yf.Ticker(B_ticker_name)

# Returns the environment, social and governance score, like on yahoo finance
def get_esg_risk(ticker):
    data_points = ["environmentScore", "socialScore", "governanceScore"]
    data = ticker.sustainability
    try:
        return data.loc[data_points].transpose()
    except:
        #print('Could not get ESG score for ' + ticker.ticker)
        return None

def get_esg_risk_array(ticker):
    df = get_esg_risk(ticker)
    if df is None:
        return [None]
    return np.array(get_esg_risk(ticker).loc['Value'])

print("Sustainability (ESG)")
print(get_esg_risk_array(assetA_ticker))
print(get_esg_risk_array(assetB_ticker))


# NOTE that these assets will have dividends and splits accounted for in the historical price
assetA_close = assetA_ticker.history(start=start)["Close"].fillna(method="ffill")
assetB_close = assetB_ticker.history(start=start)["Close"].fillna(method="ffill")
raw_A = assetA_close

# Normalize so that all start at 1
assetA_close /= float(assetA_close[:1])
assetB_close /= float(assetB_close[:1])
GSPC /= float(GSPC[:1])


# Takes in normalized time series that starts with 1
# regression with exponential curve
def exp_reg_curve(input_series):
    reg_vals = np.polyfit(range(len(input_series.index)), np.log(input_series), 1, w=np.sqrt(input_series))
    
    def exp_func(x):
        y = np.exp(reg_vals[1]) * np.exp(reg_vals[0] * x)
        return y

    return pd.Series(data=exp_func(np.array(range(len(input_series.index)))), index=input_series.index)

# Calculates deviation from exponential regression model and normalizes by dividing each deviation by the exp_model_value
# Takes in normalized time series that starts with 1
def dev_curve_from_exp_normd(input_series):
    reg_input_series = exp_reg_curve(input_series)
    dev_input_series_from_exp_norm = (input_series - reg_input_series) / reg_input_series
    return dev_input_series_from_exp_norm


# Get CAGR of exp-model and a volatility measure from a time series
def get_exp_cagr_and_vol(input_series):
    # Do exp regression on input_series
    reg_vals = np.polyfit(range(len(input_series.index)), np.log(input_series), 1, w=np.sqrt(input_series))

    # get growth rate from exp model
    cagr = np.exp(reg_vals[0]) ** 253 - 1

    # get volatility compared to market volatility (S&P500) during the period of the input_series
    market_series = yf.Ticker("^GSPC").history(start=input_series.index[0], end=input_series.index[-1])["Close"].fillna(method="ffill")
    market_series /= float(market_series[:1])

    market_std = dev_curve_from_exp_normd(market_series).std()
    volatility = dev_curve_from_exp_normd(input_series).std() / market_std

    return [cagr, volatility]

# print some data about the assets
print("CAGR")
print(get_exp_cagr_and_vol(assetA_close)[0])
print(get_exp_cagr_and_vol(assetB_close)[0])
print("VOLATILITY")
print(get_exp_cagr_and_vol(assetA_close)[1])
print(get_exp_cagr_and_vol(assetB_close)[1])


fig, ax = plt.subplots()
ax.plot(assetA_close, label=A_ticker_name)
ax.plot(assetB_close, label=B_ticker_name)
ax.plot(GSPC, label="GSPC")
ax.plot(exp_reg_curve(GSPC))
ax.plot(exp_reg_curve(assetA_close))
ax.plot(exp_reg_curve(assetB_close))
ax.legend()

ax.set(xlabel='Date', ylabel='Index',
       title='Stock Market Overview')
ax.grid()

#plt.yscale('log')
plt.show()
