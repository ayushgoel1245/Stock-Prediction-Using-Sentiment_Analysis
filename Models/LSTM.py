import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Normalize stock prices
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

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=16, epochs=10)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(np.hstack((train_predict, np.zeros((train_predict.shape[0], 1)))))
test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 1)))))

# Calculate RMSE and MAE
train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict[:, 0]))

train_mae = mean_absolute_error(Y_train, train_predict[:, 0])
test_mae = mean_absolute_error(Y_test, test_predict[:, 0])

# Print RMSE and MAE
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'Train MAE: {train_mae}')
print(f'Test MAE: {test_mae}')

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(merged_data['Date'], merged_data['Close'], label='Actual Prices', color='blue')
plt.plot(merged_data['Date'][:len(train_predict)], train_predict[:, 0], label='Train Predictions', color='green')
plt.plot(merged_data['Date'][len(train_predict):len(train_predict) + len(test_predict)], test_predict[:, 0], label='Test Predictions', color='red')
plt.legend()
plt.title('Stock Price Prediction with Sentiment Using LSTM ')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
