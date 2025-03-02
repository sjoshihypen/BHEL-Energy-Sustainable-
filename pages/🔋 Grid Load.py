import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Set Page Config
st.set_page_config(page_title="AI Renewable Energy Forecasting", layout="wide")

# Load the new uploaded BHEL logo
bhel_logo_path = "D:/BHEL/Images/bhel.jpg"

# Title and Header with BHEL image on the right
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='font-weight: bold;'>🔋 Load Balancing AI Grid &<br>Demand Prediction</h1>", unsafe_allow_html=True)

with col2:
    st.image(bhel_logo_path, use_container_width=True)  # Fix: Use 'use_container_width' instead of 'use_column_width'


# Introduction
st.markdown("""
Welcome to the AI-powered dashboard designed for BHEL engineers. This tool leverages deep learning models to predict real-time energy demand and facilitates dynamic power redistribution, ensuring grid stability and optimized energy distribution.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")
def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', -10, 50, 25)
    humidity = st.sidebar.slider('Humidity (%)', 0, 100, 50)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 0, 20, 5)
    hour_of_day = st.sidebar.slider('Hour of Day', 0, 23, 12)
    day_of_week = st.sidebar.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    data = {
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        file_path = "D:/BHEL/Dataset/historical_energy_consumption.csv"
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        data = pd.read_csv(file_path)
        if 'timestamp' not in data.columns:
            raise ValueError("The 'timestamp' column is missing from the CSV file.")
        
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.dropna(subset=['timestamp'])
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

if not data.empty:
    data['hour_of_day'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek

    def preprocess_data(data):
        features = ['temperature', 'humidity', 'wind_speed', 'hour_of_day', 'day_of_week']
        target = 'demand'

        if any(feat not in data.columns for feat in features):
            st.error("Missing required feature columns.")
            return None, None, None

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[features])
        return scaled_features, data[target].values, scaler

    X, y, scaler = preprocess_data(data)

    if X is not None and y is not None:
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        def load_or_train_model(X, y):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            model_path = 'lstm_model.weights.h5'  # Fixed naming issue

            if os.path.exists(model_path):
                model.load_weights(model_path)
            else:
                model.fit(X, y, batch_size=1, epochs=10)
                model.save_weights(model_path)

            return model

        model = load_or_train_model(X, y)

        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        input_df['day_of_week'] = input_df['day_of_week'].map(day_mapping)

        def predict_demand(input_data, model, scaler):
            try:
                if input_data.empty:
                    raise ValueError("Input data is empty. Please check input values.")
                
                input_data_scaled = scaler.transform(input_data)
                input_data_scaled = np.reshape(input_data_scaled, (input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
                prediction = model.predict(input_data_scaled)
                return prediction[0][0]
            except Exception as e:
                st.error(f"Error in prediction: {e}")
                return None

        prediction = predict_demand(input_df, model, scaler)
        if prediction is not None:
            st.subheader('⚡ Predicted Energy Demand (MW)')
            st.success(f"{prediction:.2f} MW")

        # Historical Data Visualization
        st.subheader('📊 Historical Energy Demand')
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data['timestamp'], y=data['demand'], mode='lines', name='Historical Demand'))
        fig1.update_layout(title='Energy Demand Over Time', xaxis_title='Time', yaxis_title='Demand (MW)')
        st.plotly_chart(fig1, use_container_width=True)

else:
    st.warning("No data available to display.")
