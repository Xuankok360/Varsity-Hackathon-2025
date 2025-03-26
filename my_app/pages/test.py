import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import seaborn as sns
#& "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run "C:\Users\user\OneDrive - Asia Pacific University\Data\FYP\my_app\home.py"

# ------------------------------------------------------------------------
# 1) Page Title & Config
# ------------------------------------------------------------------------
st.title("Test Model & View Trade Performance")

LATESTDATA_FOLDER = r"app\Interval"
MODEL_FOLDER = r"app\Model"
TRADES_SAVE_FOLDER = r"app\Trade"

# ------------------------------------------------------------------------
# 2) Let the user pick the test CSV & model
# ------------------------------------------------------------------------
available_csv = [f for f in os.listdir(LATESTDATA_FOLDER) if f.endswith('.csv')]
if not available_csv:
    st.error("No CSV files found in the 'latestdata' folder. Add CSVs and reload.")
    st.stop()

selected_csv = st.selectbox("Select CSV file for testing:", available_csv)

available_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.h5')]
if not available_models:
    st.error("No .h5 model files found in the 'Model' folder.")
    st.stop()

selected_model = st.selectbox("Select model to load:", available_models)

if st.button("Run Test & Generate Report"):

    # --------------------------------------------------------------------
    # 3) LOAD & PREPARE THE DATA
    # --------------------------------------------------------------------
    data_path = os.path.join(LATESTDATA_FOLDER, selected_csv)
    st.write(f"Loading test CSV: `{data_path}`")

    df = pd.read_csv(data_path, parse_dates=['date'])
    st.subheader("Preview of Selected CSV File")
    st.dataframe(df.head())  # Display the first few rows

    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=['date'], inplace=True)  # remove any duplicate dates

    # Define the feature columns needed
    features = ['open', 'high', 'low', 'close', 'volume', 'oi', 'coinbase_premium_index']

    # Convert to numeric
    for col in features:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with invalid close
    df = df[df['close'] > 0]

    df['return'] = df['close'].pct_change()
    df['future_return'] = df['return'].shift(-1)
    df['target'] = (df['future_return'] > 0).astype(int)
    df.dropna(subset=features + ['target'], inplace=True)

    # --------------------------------------------------------------------
    # 4) CREATE SEQUENCES FOR THE CNN
    # --------------------------------------------------------------------
    window_size = 10

    def create_sequences(data, feature_cols, window_size):
        X, y, dates = [], [], []
        for i in range(window_size, len(data)):
            X.append(data[feature_cols].iloc[i - window_size:i].values)
            y.append(data['target'].iloc[i])
            dates.append(data['date'].iloc[i])
        return np.array(X), np.array(y), np.array(dates)

    train_size = 0.8
    split_idx = int(len(df) * train_size)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    X_train_seq, y_train_seq, dates_train = create_sequences(df_train, features, window_size)
    X_test_seq,  y_test_seq,  dates_test  = create_sequences(df_test, features, window_size)

    # --------------------------------------------------------------------
    # 5) SCALE THE DATA
    # --------------------------------------------------------------------
    scaler = StandardScaler()
    num_features = len(features)

    X_train_2d = X_train_seq.reshape(-1, num_features)
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_train_seq_scaled = X_train_scaled_2d.reshape(X_train_seq.shape)

    X_test_2d = X_test_seq.reshape(-1, num_features)
    X_test_scaled_2d = scaler.transform(X_test_2d)
    X_test_seq_scaled = X_test_scaled_2d.reshape(X_test_seq.shape)

    # --------------------------------------------------------------------
    # 6) LOAD MODEL & MAKE PREDICTIONS
    # --------------------------------------------------------------------
    model_path = os.path.join(MODEL_FOLDER, selected_model)
    st.write(f"Loading model: `{model_path}`")

    model = load_model(model_path)
    predictions = (model.predict(X_test_seq_scaled) > 0.5).astype(int).flatten()

    df_test_sim = df_test.iloc[window_size:].copy().reset_index(drop=True)
    if len(df_test_sim) != len(predictions):
        min_len = min(len(df_test_sim), len(predictions))
        df_test_sim = df_test_sim.iloc[:min_len].copy()
        predictions = predictions[:min_len]
    df_test_sim['prediction'] = predictions

    # --------------------------------------------------------------------
    # 7) BACKTEST SIMULATION (EXCLUDING FEES)
    # --------------------------------------------------------------------
    initial_balance = 100000.0
    balance = initial_balance
    position = None
    entry_price = 0.0
    trades = []

    for i in range(len(df_test_sim) - 1):
        current_row = df_test_sim.iloc[i]
        next_row = df_test_sim.iloc[i + 1]

        current_date = current_row['date']
        next_date = next_row['date']
        current_price = current_row['close']
        next_price = next_row['close']
        current_signal = current_row['prediction']
        next_signal = next_row['prediction']

        # If no position, open one based on the current signal.
        if position is None:
            if current_signal == 1:
                position = 'long'
                entry_price = current_price
                entry_date = current_date
            else:
                position = 'short'
                entry_price = current_price
                entry_date = current_date
        else:
            if position == 'long' and next_signal == 0:
                exit_price = next_price
                exit_date = next_date
                pnl = (exit_price - entry_price)
                balance += pnl
                result = 'win' if pnl >= 0 else 'loss'
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': result,
                    'balance': balance,
                    'type': 'long_exit'
                })
                # Open new short position at exit.
                position = 'short'
                entry_price = exit_price
                entry_date = exit_date
            elif position == 'short' and next_signal == 1:
                exit_price = next_price
                exit_date = next_date
                pnl = (entry_price - exit_price)
                balance += pnl
                result = 'win' if pnl >= 0 else 'loss'
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': result,
                    'balance': balance,
                    'type': 'short_exit'
                })
                # Open new long position at exit.
                position = 'long'
                entry_price = exit_price
                entry_date = exit_date

    df_trades = pd.DataFrame(trades, columns=[
        'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl',
        'result', 'balance', 'type'
    ])

    # --------------------------------------------------------------------
    # 8) CONSTRUCT EQUITY CURVE
    # --------------------------------------------------------------------
    df_test_sim['equity'] = np.nan
    eq = initial_balance
    trade_idx = 0
    current_trade = None

    for i in range(len(df_test_sim)):
        date_i = df_test_sim.loc[i, 'date']
        df_test_sim.loc[i, 'equity'] = eq
        if trade_idx < len(df_trades):
            if current_trade is None:
                # Check if a new trade starts here
                if pd.to_datetime(df_trades.loc[trade_idx, 'entry_date']) == date_i:
                    current_trade = df_trades.iloc[trade_idx]
            else:
                # Check if that trade ends here
                if pd.to_datetime(current_trade['exit_date']) == date_i:
                    eq = current_trade['balance']
                    df_test_sim.loc[i, 'equity'] = eq
                    trade_idx += 1
                    current_trade = None
                    if trade_idx < len(df_trades):
                        nxt_entry_date = pd.to_datetime(df_trades.loc[trade_idx, 'entry_date'])
                        if nxt_entry_date == date_i:
                            current_trade = df_trades.iloc[trade_idx]

    df_test_sim['equity'].fillna(method='ffill', inplace=True)
    df_test_sim.dropna(inplace=True)

    # --------------------------------------------------------------------
    # 9) PERFORMANCE METRICS
    # --------------------------------------------------------------------
    df_test_sim['strategy_return'] = df_test_sim['equity'].pct_change().fillna(0)
    # For ~4h data => 6 * 365 = 2190 periods/year
    periods_per_year = 6 * 365
    mean_return = df_test_sim['strategy_return'].mean() * periods_per_year
    std_return = df_test_sim['strategy_return'].std() * np.sqrt(periods_per_year)
    sharpe_ratio = mean_return / (std_return + 1e-9)

    roll_max = df_test_sim['equity'].cummax()
    drawdown = (df_test_sim['equity'] - roll_max) / roll_max
    max_drawdown = drawdown.min()

    # --------------------------------------------------------------------
    # 10) PLOT RESULTS
    # --------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.set_title("BTC/USD Close Price & Equity Curve")
    ax1.set_xlabel("Date")

    # Plot close price
    ax1.plot(df_test_sim['date'], df_test_sim['close'], label='Close Price', alpha=0.7)
    ax1.set_ylabel("Close Price")

    # Plot equity on a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(df_test_sim['date'], df_test_sim['equity'], label='Equity', alpha=0.7, color='orange')
    ax2.set_ylabel("Equity")

    # Plot trade entry markers on close price
    for idx, row in df_trades.iterrows():
        entry_date = pd.to_datetime(row['entry_date'])
        entry_px   = row['entry_price']
        if 'long' in row['type']:
            ax1.plot(entry_date, entry_px, '^', markersize=8, color='green')
        else:
            ax1.plot(entry_date, entry_px, 'v', markersize=8, color='red')

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    st.pyplot(fig)

    # --------------------------------------------------------------------
    # 11) SAVE TRADES TO CSV
    # --------------------------------------------------------------------
    trades_output_path = os.path.join(TRADES_SAVE_FOLDER, "trades_output.csv")
    df_trades.to_csv(trades_output_path, index=False)
    st.success(f"Trades saved to {trades_output_path}")

    # Let the user download the trade history directly
    csv_data = df_trades.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Trade History CSV",
        data=csv_data,
        file_name="trade_history.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------------------
    # 12) PRINT TRADE PERFORMANCE
    # --------------------------------------------------------------------
    start_date = df_trades['entry_date'].iloc[0] if not df_trades.empty else None
    end_date   = df_trades['exit_date'].iloc[-1] if not df_trades.empty else None
    total_trades = len(df_trades)
    final_balance = df_trades['balance'].iloc[-1] if not df_trades.empty else initial_balance

    st.write("---")
    st.write("**Performance Summary**")
    st.write(f"- Sharpe Ratio: `{sharpe_ratio:.3f}`")
    st.write(f"- Max Drawdown: `{max_drawdown:.2%}`")
    st.write(f"- Trades Count: `{total_trades}`")
    if start_date and end_date:
        st.write(f"- Test Start: `{start_date}`")
        st.write(f"- Test End: `{end_date}`")
    st.write(f"- Final Balance: `{final_balance:,.2f}`")

    st.write("**First 10 Trades**")
    st.dataframe(df_trades.head(10))

    # --------------------------------------------------------------------
    # 13) DISPLAY CORRELATION MATRIX OF RAW DATA FEATURES
    # --------------------------------------------------------------------
    st.write("---")
    st.write("**Correlation Matrix of Raw Data Features**")
    raw_corr = df[features].corr()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(raw_corr, annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Matrix of Raw Data Features")
    st.pyplot(fig2)
