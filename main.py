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

# 365 days - calculate at the level of one year
DAYS = 365


# Read the input data from csv file, get around the weird date format,
# set the index and sort by time
df = pd.read_csv("resources/DailyBitcoinPriceEvolution.csv")
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.set_index(['Date'], inplace=True)
df.sort_index(inplace=True)

# Plot the time series
plt.plot(df, color='orange')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Daily Bitcoin Price (2016-2021)', fontsize=15)
plt.show()

# Begin transforming the series to make it stationary
# Get the values only as a pandas series and apply dickie fuller
test_stationarity(df['Price'], DAYS)

# Log transform the series, plot and apply again
ts_log = np.log(df.replace(0, np.nan))

# Drop NaN values and test
ts_log = ts_log[ts_log['Price'].notna()]
test_stationarity(ts_log['Price'], DAYS)

# Remove trend and seasonality with differencing, plot, drop NaN values
#  and apply again
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff, label='Difference', color='green')
plt.legend(loc='best')
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff['Price'], DAYS)


# Decompose data into components
decomposition = seasonal_decompose(ts_log, model='additive', period=DAYS)
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

# Test stationarity of residuals (noise)
plt.plot(residual, label='Residuals (noise)', color='blue')
plt.legend(loc='best')
plt.show()
residual.dropna(inplace=True)
test_stationarity(residual, DAYS)
plt.show()

# Plot ACF and PACF to find p, d, q - smoothen using rolling mean
log_rollmean = ts_log.rolling(window=DAYS, center=False).mean()
log_rollmean.dropna(inplace=True)
num_rows, num_columns = log_rollmean.shape
lag_acf = acf(log_rollmean, nlags=num_rows)
lag_pacf = pacf(log_rollmean, nlags=15, method='ols')
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
model = ARIMA(ts_log['Price'], order=(1, 1, 0))
results_AR = model.fit(disp=-1)
print(results_AR.fittedvalues - ts_log_diff['Price'])
plt.plot(ts_log_diff, color='blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_AR.fittedvalues, ts_log_diff['Price']))
plt.show()

# Moving Average Model
model = ARIMA(ts_log['Price'], order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_MA.fittedvalues, ts_log_diff['Price']))
plt.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log['Price'], order=(2, 1, 0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_ARIMA.fittedvalues, ts_log_diff['Price']))
plt.show()

# Divide into train and test datasets
size = int(len(ts_log) - 100)
train_arima, test_arima = ts_log['Price'][0:size], ts_log['Price'][size:len(
    ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Predicted vs Expected Values\n')
# We go over each value in the test set and then apply ARIMA model and calculate the predicted value.
# We have the expected value in the test set therefore we calculate the error between predicted and expected value
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)

    output = model_fit.forecast()

    pred_value = output[0]

    original_value = test_arima[t]
    history.append(original_value)

    pred_value = np.exp(pred_value)

    original_value = np.exp(original_value)

    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' %
          (pred_value, original_value, error), '%')

    predictions.append(float(pred_value))
    originals.append(float(original_value))

# After iterating over whole test set the overall mean error is calculated.
print('\n Mean Error in Predicting Test Case Articles : %f ' %
      (sum(error_list) / float(len(error_list))), '%')
plt.figure(figsize=(8, 6))
test_day = [t for t in range(len(test_arima))]
plt.plot(test_day, predictions, color='red')
plt.plot(test_day, originals, color='green')
plt.title('Expected Vs Predicted value of Bitcoin')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend({'Original', 'Predicted'})
plt.show()

# Now we can forecast values
results_ARIMA.plot_predict(1, len(ts_log['Price']) + 100)
plt.show()

x_vals = [x for x in range(len(ts_log['Price']) + 100)]
y_vals = [np.exp(x) for x in ts_log['Price']]
y_vals2 = [np.nan for x in ts_log['Price']]
for x in results_ARIMA.forecast(100)[0]:
    y_vals.append(np.nan)
    y_vals2.append(np.exp(x))
plt.plot(x_vals, y_vals, label='Original', color='red')
plt.plot(x_vals, y_vals2, label='Predicted', color='green')
plt.show()
