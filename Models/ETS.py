from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prepare ETS data
ets_data = merged_data.set_index('Date')[['Close', 'Sentiment']]

# Split data into training and testing sets
train_size = int(len(ets_data) * 0.8)
ets_train, ets_test = ets_data.iloc[:train_size], ets_data.iloc[train_size:]

# ETS model: Using only stock prices (sentiment could be indirectly incorporated using SARIMAX)
ets_model = ExponentialSmoothing(
    ets_train['Close'],
    trend='add',
    seasonal='add',
    seasonal_periods=12  # Assuming monthly seasonality for demonstration
).fit()

# Predictions
ets_train_pred = ets_model.fittedvalues
ets_test_pred = ets_model.forecast(steps=len(ets_test))

# Calculate error metrics
ets_rmse = np.sqrt(((ets_test['Close'] - ets_test_pred) ** 2).mean())
ets_mae = np.abs(ets_test['Close'] - ets_test_pred).mean()

# Plot ETS results
plt.figure(figsize=(12, 6))
plt.plot(ets_data.index, ets_data['Close'], label='Actual Prices', color='blue')
plt.plot(ets_train.index, ets_train_pred, label='ETS Train Predictions', color='green')
plt.plot(ets_test.index, ets_test_pred, label='ETS Test Predictions', color='red')
plt.legend()
plt.title('ETS Stock Price Prediction with Sentiment')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

print(f"ETS RMSE: {ets_rmse}, ETS MAE: {ets_mae}")
