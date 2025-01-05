import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility functions for indicators
def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except ZeroDivisionError:
        logging.warning("Division by zero in RSI calculation.")
        return pd.Series([0] * len(series))

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = calculate_ema(series, short_period)
    long_ema = calculate_ema(series, long_period)
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal_line

def calculate_bollinger_bands(series, period=20):
    sma = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    return upper_band, lower_band

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(window=period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(window=period).mean()

def calculate_fibonacci_retracement(high, low, close):
    max_price = high.max()
    min_price = low.min()
    diff = max_price - min_price
    levels = {
        '0.236': max_price - diff * 0.236,
        '0.382': max_price - diff * 0.382,
        '0.5': max_price - diff * 0.5,
        '0.618': max_price - diff * 0.618,
        '0.786': max_price - diff * 0.786
    }
    return levels

# Add indicators to the dataset
def add_indicators(df):
    try:
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['EMA_20'] = calculate_ema(df['Close'], period=20)
        df['MACD'] = calculate_macd(df['Close'])
        df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
        fib_levels = calculate_fibonacci_retracement(df['High'], df['Low'], df['Close'])
        df['Fib_0.236'] = fib_levels['0.236']
        df['Fib_0.382'] = fib_levels['0.382']
        df['Fib_0.5'] = fib_levels['0.5']
        df['Fib_0.618'] = fib_levels['0.618']
        df['Fib_0.786'] = fib_levels['0.786']
        df = df.fillna(0)  # Fill missing values
    except Exception as e:
        logging.error(f"Error in adding indicators: {e}")
    return df

# File loading with error handling
def load_and_preprocess_data(path):
    try:
        filenames = [f for f in os.listdir(path) if f.endswith('.csv')]
        dataframes = [pd.read_csv(os.path.join(path, filename)) for filename in filenames]
        combined_df = pd.concat(dataframes, ignore_index=True)

        if 'Date' in combined_df.columns:
            combined_df = combined_df.drop('Date', axis=1)

        combined_df = add_indicators(combined_df)
        return combined_df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Neural Network Model
def build_model(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Enhanced Backtesting with risk-reward and transaction costs
def backtest_model(data, model, risk_reward_ratio=2, transaction_cost=0.01):
    wallet = 1000
    results = []

    data = pd.DataFrame(data, columns=['Close', 'High', 'Low'])
    predictions = model.predict(data.drop(columns=['Close'], axis=1), batch_size=128)
    for index, (row, prediction) in enumerate(zip(data.iterrows(), predictions)):
        row = row[1]  # Extract the actual row
        if prediction > 0.5:
            buy_price = row['Close']
            target = buy_price * (1 + (0.01 * risk_reward_ratio))
            stop_loss = buy_price * (1 - 0.01)

            # Simplified logic
            trade_outcome = (row['High'] >= target) - (row['Low'] <= stop_loss)
            profit = (target - buy_price) if trade_outcome > 0 else (buy_price - stop_loss)
            wallet += profit - transaction_cost
            results.append(trade_outcome > 0)

    return results, wallet

# Training and evaluation
folder_path = 'D:\\om\\Training Model\\Layer1\\Now1'
data = load_and_preprocess_data(folder_path)

if not data.empty:
    # Prepare features and target
    X = data.drop(['Close'], axis=1).reset_index(drop=True)
    Y = (data['Close'].shift(-1) > data['Close']).astype(int).reset_index(drop=True)

    # Remove the last row to avoid NaN target
    X = X[:-1]
    Y = Y[:-1]

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Reset indices to avoid mismatched indexing issues
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # Convert data to NumPy arrays for TensorFlow compatibility
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Check for NaNs and fill them
    if np.isnan(X_train).sum() > 0 or np.isnan(X_test).sum() > 0:
        logging.warning("Filling NaN values in training/testing data.")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

    # Debugging: Log data shapes and ensure correctness
    logging.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")


    # Build and compile the model
    model = build_model(X_train.shape[1])
    checkpoint_path = 'model_checkpoint.keras'

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train the model
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, Y_test),
        callbacks=[
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
            lr_scheduler,
            early_stopping
        ],
        class_weight={0: 1, 1: 3}  # Adjust for imbalanced data
    )

    # Evaluate the model
    Y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred)

    logging.info(f"Model Evaluation: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, ROC_AUC={roc_auc}")
else:
    logging.error("Data loading failed or dataset is empty. Please check the input path or data format.")


