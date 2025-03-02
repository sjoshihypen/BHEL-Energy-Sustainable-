import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random

# Set Page Config
st.set_page_config(page_title="AI Renewable Energy Forecasting", layout="wide")

# Load the new uploaded BHEL logo
bhel_logo_path = "D:/BHEL/Images/bhel.jpg"

# Title and Header with BHEL image on the right
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='font-weight: bold;'>Energy Trading &<br>AI Grid Revenue Optimization</h1>", unsafe_allow_html=True)

with col2:
    st.image(bhel_logo_path, use_container_width=True)  # Fix: Use 'use_container_width' instead of 'use_column_width'


# Header
st.markdown("#### Optimize energy sales using AI-driven forecasting & smart bidding 💡")

# Simulating Electricity Market Data
def generate_market_data():
    np.random.seed(42)
    hours = np.arange(24)
    base_prices = np.sin(hours / 3) * 20 + 50  # Simulated price fluctuations
    volatility = np.random.uniform(-5, 5, 24)  # Random market variations
    prices = base_prices + volatility
    demand = np.random.randint(50, 150, 24)  # Simulated grid demand
    return pd.DataFrame({"Hour": hours, "Price": prices, "Demand": demand})

market_data = generate_market_data()

# Visualizing Electricity Prices
fig1 = px.line(market_data, x="Hour", y="Price", title="🔍 Market Electricity Price Trends")
st.plotly_chart(fig1, use_container_width=True)

# AI-Based Price Prediction
st.markdown("### 📈 AI-Powered Price Prediction")
X = market_data[["Hour", "Demand"]]
y = market_data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
st.metric(label="📊 Prediction Model Accuracy (MAE)", value=f"{mae:.2f}")

# Smart Bidding System
def smart_bidding(price, demand):
    threshold = np.percentile(market_data["Price"], 75)  # Sell at top 25% price
    if price > threshold and demand > 100:
        return "✅ Sell Now"
    elif price < threshold - 10:
        return "⚠️ Hold Energy"
    else:
        return "🔄 Monitor Market"

market_data["Trading Decision"] = market_data.apply(lambda row: smart_bidding(row.Price, row.Demand), axis=1)
fig2 = px.bar(market_data, x="Hour", y="Price", color="Trading Decision", title="📊 AI Trading Strategy")
st.plotly_chart(fig2, use_container_width=True)

# Blockchain Energy Trading Simulation
st.markdown("### 🔗 Blockchain Energy Trade Simulation")
transactions = []
for i in range(10):
    transactions.append({
        "Time": f"{random.randint(1, 24)}:00", 
        "Seller": f"BHEL Solar #{random.randint(1,5)}", 
        "Buyer": f"Grid Company #{random.randint(1,3)}", 
        "Energy (MWh)": round(random.uniform(1, 5), 2),
        "Price (₹/MWh)": round(random.uniform(50, 80), 2)
    })
blockchain_df = pd.DataFrame(transactions)
st.dataframe(blockchain_df)

st.success("🚀 AI-optimized energy trading can boost revenue by 15-25%! Integrate this into BHEL's grid now.")
