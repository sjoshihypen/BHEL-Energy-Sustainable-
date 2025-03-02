import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="AI Renewable Energy Forecasting", layout="wide")

# Load the new uploaded BHEL logo
bhel_logo_path = "D:/BHEL/Images/bhel.jpg"

# Title and Header with BHEL image on the right
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='font-weight: bold;'>Weather AI Impact Analysis &<br>Disaster Preparedness for Renewable Plants</h1>", unsafe_allow_html=True)

with col2:
    st.image(bhel_logo_path, use_container_width=True)  # Fix: Use 'use_container_width' instead of 'use_column_width'

# Load historical weather and damage data
def load_data():
    data = pd.read_csv('historical_weather_damage_data.csv')
    return data

# Train an AI model to predict damage risk
def train_model(data):
    features = data[['wind_speed', 'precipitation', 'temperature']]
    target = data['damage_occurred']
    model = RandomForestClassifier()
    model.fit(features, target)
    return model

# Predict damage risk based on upcoming weather conditions
def predict_risk(model, upcoming_weather):
    prediction = model.predict(upcoming_weather)
    risk_level = 'High' if prediction == 1 else 'Low'
    return risk_level

# Visualize historical data
def visualize_data(data):
    fig = px.scatter(data, x='wind_speed', y='damage_cost', color='damage_occurred',
                     title='Historical Damage Data')
    st.plotly_chart(fig, use_container_width=True)

# Main function to run the Streamlit app
def main():
    st.subheader('Historical Weather and Damage Data')
    data = load_data()
    st.write(data.head())

    # Train model
    model = train_model(data)

    # Input upcoming weather conditions
    st.subheader('Upcoming Weather Conditions')
    wind_speed = st.number_input('Wind Speed (km/h)', min_value=0)
    precipitation = st.number_input('Precipitation (mm)', min_value=0)
    temperature = st.number_input('Temperature (°C)', min_value=-50, max_value=50)

    upcoming_weather = pd.DataFrame([[wind_speed, precipitation, temperature]],
                                    columns=['wind_speed', 'precipitation', 'temperature'])

    # Predict and display risk level
    if st.button('Predict Damage Risk'):
        risk_level = predict_risk(model, upcoming_weather)
        st.write(f'Predicted Damage Risk Level: {risk_level}')

    # Visualize historical data
    st.subheader('Data Visualization')
    visualize_data(data)

if __name__ == '__main__':
    main()
