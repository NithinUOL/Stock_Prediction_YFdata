import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objs as go
from datetime import datetime

# Page Configuration
st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your stock data CSV", type="csv")

# Load model (assumes a model file named 'model.h5' is in the same directory)
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Fill or drop NA as appropriate
    df.dropna(inplace=True)

    # Normalize for LSTM
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    # Train-test split
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    # Create sequences
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_test, y_test = create_dataset(test_data)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Prediction
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    actual_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluation
    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{np.sqrt(mean_squared_error(actual_stock_price, predicted_stock_price)):.4f}")
    col2.metric("MAE", f"{mean_absolute_error(actual_stock_price, predicted_stock_price):.4f}")
    col3.metric("R² Score", f"{r2_score(actual_stock_price, predicted_stock_price):.4f}")

    # Plot: Actual vs Predicted
    st.subheader("📈 Actual vs Predicted Stock Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual_stock_price.flatten(), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=predicted_stock_price.flatten(), mode='lines', name='Predicted'))
    fig.update_layout(height=500, width=1000)
    st.plotly_chart(fig)

    # Show correlation heatmap
    st.subheader("🔍 Correlation Matrix")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
else:
    st.info("Please upload a CSV file to proceed.")
