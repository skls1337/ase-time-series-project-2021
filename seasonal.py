import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from library.utils import test_stationarity, rss

import warnings
from numpy.linalg.linalg import LinAlgError
warnings.filterwarnings("ignore")

MONTHS = 12

df = pd.read_csv("resources/MonthlyAverageNetWagePerTotalEconomy.csv")
df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
df.set_index(['Month'], inplace=True)
df.sort_index(inplace=True)
print(df.head(10))
print(type(df.index))

# Plot the time series
plt.plot(df, color='blue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Wage', fontsize=12)
plt.title('Averge Monthly Wage Earned in Romania, 2005-2021', fontsize=15)
plt.show()

test_stationarity(df['Salary'], MONTHS)

# Log transform the series, plot and apply again
ts_log = np.log(df.replace(0, np.nan))

# Drop NaN values and test
ts_log = ts_log[ts_log['Salary'].notna()]
plt.plot(ts_log, label='Log-transformed', color='green')
plt.legend(loc='best')
plt.show()
ts_log.dropna(inplace=True)
test_stationarity(ts_log['Salary'], MONTHS)

# Remove trend and seasonality with differencing, plot, drop NaN values
#  and apply again
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff, label='Difference', color='green')
plt.legend(loc='best')
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff['Salary'], MONTHS)

# Decompose data into components
decomposition = seasonal_decompose(
    ts_log, model='multiplicative', period=MONTHS)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(ts_log, label='Original', color='blue')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', color='blue')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='blue')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals', color='blue')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Plot ACF and PACF to find p, d, q - smoothen using rolling mean
log_rollmean = ts_log.rolling(window=MONTHS, center=False).mean()
log_rollmean.dropna(inplace=True)
lag_acf = acf(log_rollmean, nlags=100)
lag_pacf = pacf(log_rollmean, nlags=20, method='ols')
print(lag_acf)
print(lag_pacf)

# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf, color='blue')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(
    y=-1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.axhline(
    y=1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.title('Autocorrelation Graph')

# Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf, color='blue')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Graph')
plt.tight_layout()
plt.show()

# Auto Regressive model
model = ARIMA(ts_log['Salary'], order=(0, 1, 2))
results_AR = model.fit(disp=-1)
print(results_AR.fittedvalues - ts_log_diff['Salary'])
plt.plot(ts_log_diff, color='blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_AR.fittedvalues, ts_log_diff['Salary']))
plt.show()

# Moving Average Model
model = ARIMA(ts_log['Salary'], order=(2, 1, 0))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_MA.fittedvalues, ts_log_diff['Salary']))
plt.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log['Salary'], order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_ARIMA.fittedvalues, ts_log_diff['Salary']))
plt.show()

# Now we can forecast values
print(ts_log.head(10))
results_ARIMA.plot_predict(1, len(ts_log['Salary']) + 264)
plt.show()

x_vals = [x for x in range(len(ts_log['Salary']) + 264)]
y_vals = [np.exp(x) for x in ts_log['Salary']]
y_vals2 = [np.nan for x in ts_log['Salary']]
for x in results_ARIMA.forecast(264)[0]:
    y_vals.append(np.nan)
    y_vals2.append(np.exp(x))
plt.plot(x_vals, y_vals, label='Original', color='orange')
plt.plot(x_vals, y_vals2, label='Predicted', color='blue')
plt.show()
