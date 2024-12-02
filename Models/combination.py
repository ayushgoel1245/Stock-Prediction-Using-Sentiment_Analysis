import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from prophet import Prophet

# Data Preprocessing (Assume merged_data is already available with 'Date', 'Close', and 'Sentiment' columns)
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Fill missing values for Sentiment and Close
merged_data['Sentiment'].fillna(0, inplace=True)
merged_data['Sentiment'].replace([np.inf, -np.inf], 0, inplace=True)
merged_data['Close'].fillna(merged_data['Close'].mean(), inplace=True)

# Normalize the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged_data[['Close', 'Sentiment']])

# Prepare LSTM dataset with sentiment
def create_lstm_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :])  # Multi-feature input
        Y.append(data[i + time_step, 0])   # Stock price target
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_lstm_dataset(scaled_data, time_step)

# Split into train and test sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# ----------------------------------------------
# 1. ETS (SARIMAX) Model
sarimax_data = merged_data[['Close', 'Sentiment']]

sarimax_model = SARIMAX(
    sarimax_data['Close'],
    exog=sarimax_data['Sentiment'],
    order=(1, 1, 1),  # Adjust p, d, q as needed
    seasonal_order=(1, 1, 1, 7)  # Weekly seasonality
).fit(disp=False)

# Predictions for SARIMAX
sarimax_train_pred = sarimax_model.predict(start=sarimax_data.index[0], end=sarimax_data.index[-train_size-1], exog=sarimax_data['Sentiment'][:train_size])
sarimax_test_pred = sarimax_model.predict(start=sarimax_data.index[train_size], end=sarimax_data.index[-1], exog=sarimax_data['Sentiment'][train_size:])

# RMSE and MAE for SARIMAX
sarimax_rmse = np.sqrt(mean_squared_error(sarimax_data['Close'][train_size:], sarimax_test_pred))
sarimax_mae = mean_absolute_error(sarimax_data['Close'][train_size:], sarimax_test_pred)

# ----------------------------------------------
# 2. LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=16, epochs=10)

# Predictions for LSTM
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions for LSTM
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], 1)))))
test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 1)))))

# RMSE and MAE for LSTM
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict[:, 0]))

train_mae = mean_absolute_error(Y_train, train_predict[:, 0])
test_mae = mean_absolute_error(Y_test, test_predict[:, 0])

# ----------------------------------------------
# 3. Prophet Model
prophet_data = merged_data[['Close', 'Sentiment']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Sentiment': 'sentiment'})

prophet = Prophet()
prophet.add_regressor('sentiment')
prophet.fit(prophet_data[:train_size])

future = prophet.make_future_dataframe(periods=len(prophet_data) - train_size)
future['sentiment'] = prophet_data['sentiment']

forecast = prophet.predict(future)

# RMSE and MAE for Prophet
prophet_rmse = np.sqrt(mean_squared_error(prophet_data['y'][train_size:], forecast['yhat'][train_size:]))
prophet_mae = mean_absolute_error(prophet_data['y'][train_size:], forecast['yhat'][train_size:])

# ----------------------------------------------
# Plotting Results
plt.figure(figsize=(12, 6))

# Plot Actual Prices
plt.plot(merged_data.index, merged_data['Close'], label='Actual Prices', color='blue')

# Plot SARIMAX Predictions
plt.plot(sarimax_data.index[train_size:], sarimax_test_pred, label='SARIMAX Predictions', color='green')

# Plot LSTM Predictions
plt.plot(merged_data.index[:len(train_predict)], train_predict[:, 0], label='LSTM Train Predictions', color='orange')
plt.plot(merged_data.index[len(train_predict):len(train_predict) + len(test_predict)], test_predict[:, 0], label='LSTM Test Predictions', color='red')

# Plot Prophet Predictions
plt.plot(forecast['ds'][train_size:], forecast['yhat'][train_size:], label='Prophet Predictions', color='purple', linestyle='--')

plt.legend()
plt.title('Stock Price Prediction Comparison (SARIMAX, LSTM, Prophet)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# ----------------------------------------------
# Print Evaluation Metrics for All Models
print("SARIMAX RMSE:", sarimax_rmse)
print("SARIMAX MAE:", sarimax_mae)
print("LSTM Train RMSE:", train_rmse)
print("LSTM Test RMSE:", test_rmse)
print("LSTM Train MAE:", train_mae)
print("LSTM Test MAE:", test_mae)
print("Prophet RMSE:", prophet_rmse)
print("Prophet MAE:", prophet_mae)
