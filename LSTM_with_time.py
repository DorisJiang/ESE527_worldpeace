import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', etc., set as needed
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_data(filepath):
    """
    Load CSV data and sort by date.

    :param filepath: Path to the CSV file
    :return: A pandas DataFrame sorted by date
    """
    df = pd.read_csv(filepath)
    # If there is a "Date" column, convert it to datetime format and sort
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
    return df


def preprocess_data(data, feature='Close', window_size=60):
    """
    Data preprocessing: normalization and creation of time-series samples.

    :param data: A pandas DataFrame
    :param feature: The feature to predict (default: "Close")
    :param window_size: The size of the sliding window (default: 60 days)
    :return: Training samples (X), labels (y), and the normalization scaler
    """
    # Only use the specified feature column and reshape
    dataset = data[feature].values.reshape(-1, 1)

    # Normalize the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    # Use a sliding window to build training samples
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape input to the format required by LSTM: (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_model(input_shape):
    """
    Build an LSTM neural network model.

    :param input_shape: Shape of the input data, for example (45, 1)
    :return: A compiled Keras model
    """
    model = Sequential()
    # First LSTM layer, set return_sequences=True for stacking more LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    # Second LSTM layer, does not return sequences
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # Output layer for predicting continuous values
    model.add(Dense(units=1))

    # Compile the model using mean squared error as the loss function and Adam as the optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def plot_results(real_data, predicted_data, dates=None):
    """
    Plot a comparison of actual vs. predicted values for the test set
    (zoomed in on the test set only).

    :param real_data: Actual stock price data for the test set (array)
    :param predicted_data: Predicted stock price data from the model (array)
    :param dates: Corresponding date data for the test set (if available)
    """
    plt.figure(figsize=(10, 6))
    if dates is not None:
        plt.plot(dates, real_data, color='blue', label='Actual (Test Data)')
        plt.plot(dates, predicted_data, color='red', label='Predicted')
        plt.xlabel('Date')
    else:
        plt.plot(real_data, color='blue', label='Actual (Test Data)')
        plt.plot(predicted_data, color='red', label='Predicted')
        plt.xlabel('Time Step')
    plt.title('Test Data: Actual vs. Predicted')
    plt.ylabel('DJIA Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_combined(df, train_size, predictions):
    """
    Plot on one figure:
      - Actual training data (blue)
      - Actual testing data (green)
      - Predicted testing data (red)

    :param df: The original DataFrame containing Date, Close, etc.
    :param train_size: The index marking the split between training and testing
    :param predictions: Prediction results for the test set (array),
                        length should match the number of rows in the test set
    """
    plt.figure(figsize=(10, 6))
    if 'Date' in df.columns:
        # Training set
        plt.plot(df['Date'][:train_size], df['Close'][:train_size],
                 color='blue', label='Actual (Training)')
        # Test set
        plt.plot(df['Date'][train_size:], df['Close'][train_size:],
                 color='green', label='Actual (Future)')
        # Test set predictions
        plt.plot(df['Date'][train_size:], predictions,
                 color='red', linestyle='--', label='Predicted')
        plt.xlabel('Date')
    else:
        # If no "Date" column, plot by index
        plt.plot(range(train_size), df['Close'][:train_size],
                 color='blue', label='Actual (Training)')
        plt.plot(range(train_size, len(df)), df['Close'][train_size:],
                 color='green', label='Actual (Future)')
        plt.plot(range(train_size, len(df)), predictions,
                 color='red', linestyle='--', label='Predicted')
        plt.xlabel('Index')

    plt.title('Overall: Training vs. Future Actual vs. Predicted')
    plt.ylabel('DJIA Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Path to the data file (please modify as needed)
    filepath = 'upload_DJIA_table_no_outliers.csv'

    # Load data
    df = load_data(filepath)

    # Split the data by time: first 80% for training, last 20% for testing
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Preprocess the training data; here we only use the "Close" column
    window_size = 45
    X_train, y_train, scaler = preprocess_data(train_data, feature='Close', window_size=window_size)

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])

    # Prepare the test data
    # Include the last `window_size` days of the training set to build the test set
    total_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - window_size:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(window_size, len(inputs)):
        X_test.append(inputs[i - window_size:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Use the model to make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # If there is a "Date" column, get the corresponding dates for the test set
    if 'Date' in test_data.columns:
        test_dates = test_data['Date'].values
    else:
        test_dates = None

    # 1) Test set: compare actual vs. predicted
    plot_results(test_data['Close'].values, predictions, dates=test_dates)

    # 2) Additionally, plot a combined figure with training data, test data, and predictions
    #    Note: 'predictions' has the same length as the test data rows,
    #    so we can align it directly with df[train_size:].
    plot_combined(df, train_size, predictions)


if __name__ == '__main__':
    main()



