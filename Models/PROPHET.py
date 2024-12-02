from prophet import Prophet

# Prepare data for Prophet
prophet_data = merged_data[['Date', 'Close', 'Sentiment']].rename(columns={'Date': 'ds', 'Close': 'y', 'Sentiment': 'sentiment'})

# Initialize Prophet model with additional seasonalities
prophet = Prophet(
    daily_seasonality=False,  # Disable default daily seasonality
    yearly_seasonality=True,  # Enable yearly seasonality
    weekly_seasonality=True   # Enable weekly seasonality
)

# Add a custom seasonality (e.g., quarterly)
prophet.add_seasonality(name='quarterly', period=90.5, fourier_order=8)
prophet.add_regressor('sentiment')

# Fit the model
train_size = int(len(prophet_data) * 0.8)
prophet.fit(prophet_data.iloc[:train_size])

# Create future dataframe
future = prophet.make_future_dataframe(periods=len(prophet_data) - train_size)
future['sentiment'] = prophet_data['sentiment']

# Forecast
forecast = prophet.predict(future)

# Evaluate model
prophet_test = forecast.iloc[train_size:]
actual_test = prophet_data.iloc[train_size:]['y']
prophet_rmse = np.sqrt(mean_squared_error(actual_test, prophet_test['yhat']))
prophet_mae = mean_absolute_error(actual_test, prophet_test['yhat'])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual Prices', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Predictions', color='orange', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
plt.legend()
plt.title('Advanced Prophet Stock Price Prediction with Sentiment')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

print(f"Prophet RMSE: {prophet_rmse}, Prophet MAE: {prophet_mae}")
