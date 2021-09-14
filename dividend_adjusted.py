from pandas.core.frame import DataFrame
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math


###  ===================================== FUNCTIONS =========================================


### ____________________________ EXPONENTIAL REGRESSION STUFF _______________________________
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


### ___________________________________________ ESG _____________________________________________________
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

### ________________________________________ Ticker ______________________________________________
# returns a series with aggregated dividends for each year in the format that year-01-01 will contain the sum of 
# dividends for that year
def get_aggregated_dividends_per_year(ticker):
    dividends = ticker.dividends

    agg_divs = dividends.groupby(dividends.index.year).sum()
    agg_divs.index = pd.to_datetime(agg_divs.index, format="%Y")
    return agg_divs

# returns dividend_yield for each year of dividend, date will be year-01-01, dividend is all dividend for the year
# and will be divided by the mean close price during that year 
def get_dividend_yield_per_year(ticker):
    agg_divs = get_aggregated_dividends_per_year(ticker)

    close = ticker.history(period="max")["Close"]

    # get yield for each year by dividing dividend by mean close price that year
    avg_close = close.groupby(close.index.year).mean()
    avg_close.index = pd.to_datetime(avg_close.index, format="%Y")

    yield_divs = agg_divs.copy()
    for i in agg_divs.index:
        yield_divs[i] /= avg_close[i]

    yield_divs.name = "Div. Yields"

    return yield_divs

#### ======================================= SCRIPTS ==============================================


start = "1980-01-01"
A_ticker_name = "HSBA.L"
B_ticker_name = "HSBC"
assetA_ticker = yf.Ticker(A_ticker_name)
assetB_ticker = yf.Ticker(B_ticker_name)
GSPC = yf.Ticker("^GSPC").history(start=start, end="2100-01-01")["Close"].fillna(method="ffill")

# NOTE that these assets will have dividends and splits accounted for in the historical price
assetA_close = assetA_ticker.history(start=start)["Close"].fillna(method="ffill")
assetB_close = assetB_ticker.history(start=start)["Close"].fillna(method="ffill")

assetA_close_norm = assetA_close.copy()
assetB_close_norm = assetB_close.copy()
# Normalize so that all start at 1
assetA_close_norm /= float(assetA_close[:1])
assetB_close_norm /= float(assetB_close[:1])
GSPC /= float(GSPC[:1])


agg_divs_A = get_aggregated_dividends_per_year(assetA_ticker)
agg_divs_A.index = pd.to_datetime(agg_divs_A.index,format="%Y")

agg_divs_B = get_aggregated_dividends_per_year(assetB_ticker)
agg_divs_B.index = pd.to_datetime(agg_divs_B.index, format="%Y")

print(get_dividend_yield_per_year(assetB_ticker))

# print some data about the assets
print("CAGR")
print(get_exp_cagr_and_vol(assetA_close_norm)[0])
print(get_exp_cagr_and_vol(assetB_close_norm)[0])
print("VOLATILITY")
print(get_exp_cagr_and_vol(assetA_close_norm)[1])
print(get_exp_cagr_and_vol(assetB_close_norm)[1])
#print("Sustainability (ESG Risk)")
#print(get_esg_risk_array(assetA_ticker))
#print(get_esg_risk_array(assetB_ticker))


fig, ax = plt.subplots()
ax.plot(assetA_close_norm, label=A_ticker_name)
ax.plot(assetB_close_norm, label=B_ticker_name)
ax.plot(GSPC, label="GSPC")
ax.plot(exp_reg_curve(GSPC))
ax.plot(exp_reg_curve(assetA_close_norm))
ax.plot(exp_reg_curve(assetB_close_norm))
ax.legend()

ax.set(xlabel='Date', ylabel='Index',
       title='Stock Market Overview')
ax.grid()

fig2, ax2 = plt.subplots()
ax2.plot(get_dividend_yield_per_year(assetA_ticker) * 100, label="Yield (%) " + A_ticker_name)
ax2.plot(get_dividend_yield_per_year(assetB_ticker) * 100, label="Yield (%) " + B_ticker_name)
ax2.plot(get_aggregated_dividends_per_year(assetA_ticker), label="Dividends " + A_ticker_name)
ax2.plot(get_aggregated_dividends_per_year(assetB_ticker), label="Dividends " + B_ticker_name)
ax2.legend()


ax2.set(xlabel='Date', ylabel='Value',
       title='Dividend Overview')
ax2.grid()

#plt.yscale('log')
plt.show()
