import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set Page Config
st.set_page_config(page_title="AI Renewable Energy Forecasting", layout="wide")

# Load images
top_right_img_path = "D:/BHEL/Images/bhel.jpg"

# Title & Headers
col1, col2 = st.columns([3, 1])
col1.title("AI Renewable Energy Generation Forecasting")
col1.write("Predict solar/wind energy output using AI models (LSTM, Prophet, XGBoost).")

if os.path.exists(top_right_img_path):
    col2.image(top_right_img_path, caption="Vision for Renewable Energy", use_container_width=True)
else:
    col2.error(f"Error: Image not found at {top_right_img_path}")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Renewable Energy Data (CSV Format)", type=["csv"])

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.lower().str.strip()

            date_column = next((col for col in df.columns if "date" in col), None)
            if not date_column:
                st.error("No date column found. Ensure the dataset contains a column with 'date' in its name.")
                return None
            df[date_column] = pd.to_datetime(df[date_column])
            df.dropna(subset=[date_column], inplace=True)

            expected_columns = {
                "energy output": ["energy output", "power generation", "output"],
                "temperature": ["temperature", "temp"],
                "wind speed": ["wind speed", "wind_velocity"],
                "cloud cover": ["cloud cover", "cloudiness"],
                "solar radiation": ["solar radiation", "irradiance", "solar_irradiance"]
            }

            column_mapping = {}
            for standard_name, possible_names in expected_columns.items():
                match = next((col for col in df.columns if col in possible_names), None)
                if match:
                    column_mapping[match] = standard_name

            df = df.rename(columns=column_mapping)
            missing_cols = set(expected_columns.keys()) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}. Please upload a valid dataset.")
                return None

            return df.rename(columns={date_column: 'DateTime Stamp'})
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

data = load_data(uploaded_file)

if data is not None:
    st.write("### Preview of Uploaded Data:")
    st.dataframe(data.head())

    model_choice = st.selectbox("Select AI Model for Forecasting:", ["LSTM", "Prophet", "XGBoost"])

    if model_choice == "Prophet":
        st.subheader("📈 Forecasting with Prophet")
        
        if 'energy output' in data.columns:
            df_prophet = data.rename(columns={'DateTime Stamp': 'ds', 'energy output': 'y'})
            df_prophet.dropna(subset=['ds', 'y'], inplace=True)

            try:
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=30, freq='D')
                forecast = model.predict(future)
                fig = model.plot(forecast)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error running Prophet model: {e}")
        else:
            st.error("Dataset is missing 'energy output' column for Prophet forecasting.")

    elif model_choice == "LSTM":
        st.subheader("🔄 Forecasting with LSTM")

        df_lstm = data[['temperature', 'wind speed', 'cloud cover', 'solar radiation', 'energy output']]
        train_size = int(len(df_lstm) * 0.8)

        if train_size < 10:
            st.error("Dataset too small for LSTM training. Please upload a larger dataset.")
        else:
            train, test = df_lstm[:train_size], df_lstm[train_size:]

            def create_sequences(data, n_steps=3):
                X, y = [], []
                for i in range(len(data) - n_steps):
                    X.append(data.iloc[i:i+n_steps].values)
                    y.append(data.iloc[i+n_steps]['energy output'])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train)
            X_test, y_test = create_sequences(test)

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                st.error("Insufficient data after sequence creation. Try reducing the sequence length or increasing data size.")
            else:
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.plot(y_test, label='Actual')
                ax.plot(y_pred, label='Predicted')
                ax.legend()
                st.pyplot(fig)

    st.markdown("### 📥 Download Forecasted Data")
    csv_data = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv_data, "forecasted_data.csv", "text/csv")
