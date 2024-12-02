from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Prepare data for SARIMAX
sarimax_data = merged_data.set_index('Date')[['Close', 'Sentiment']]

# Set frequency for the datetime index (daily data)
sarimax_data = sarimax_data.asfreq('D')

# Handle missing or infinite values in Sentiment and Close columns
sarimax_data['Sentiment'].fillna(0, inplace=True)  # Fill missing sentiment values with 0
sarimax_data['Sentiment'].replace([np.inf, -np.inf], 0, inplace=True)  # Replace inf with 0

sarimax_data['Close'].fillna(sarimax_data['Close'].mean(), inplace=True)  # Fill missing close prices with the mean

# Split data into train and test sets
train_size = int(len(sarimax_data) * 0.8)
sarimax_train, sarimax_test = sarimax_data.iloc[:train_size], sarimax_data.iloc[train_size:]

# Fit SARIMAX model
sarimax_model = SARIMAX(
    sarimax_train['Close'],
    exog=sarimax_train['Sentiment'],
    order=(1, 1, 1),  # Adjust p, d, q as needed
    seasonal_order=(1, 1, 1, 7)  # Assuming weekly seasonality
).fit(disp=False)

# Predictions
sarimax_train_pred = sarimax_model.predict(
    start=sarimax_train.index[0], end=sarimax_train.index[-1], exog=sarimax_train['Sentiment']
)
sarimax_test_pred = sarimax_model.predict(
    start=sarimax_test.index[0], end=sarimax_test.index[-1], exog=sarimax_test['Sentiment']
)

# Calculate error metrics
sarimax_rmse = np.sqrt(mean_squared_error(sarimax_test['Close'], sarimax_test_pred))
sarimax_mae = mean_absolute_error(sarimax_test['Close'], sarimax_test_pred)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(sarimax_data.index, sarimax_data['Close'], label='Actual Prices', color='blue')
plt.plot(sarimax_train.index, sarimax_train_pred, label='SARIMAX Train Predictions', color='green')
plt.plot(sarimax_test.index, sarimax_test_pred, label='SARIMAX Test Predictions', color='red')
plt.legend()
plt.title('SARIMAX Stock Price Prediction with Sentiment')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

print(f"SARIMAX RMSE: {sarimax_rmse}, SARIMAX MAE: {sarimax_mae}")

