import streamlit as st
import sqlite3
import os
from PIL import Image

# Database setup
conn = sqlite3.connect('visitor_count.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS visits (
        count INTEGER
    )
''')
conn.commit()

# Initialize visitor count if not present
c.execute('SELECT count FROM visits')
row = c.fetchone()
if row is None:
    c.execute('INSERT INTO visits (count) VALUES (0)')
    conn.commit()

# Increment visitor count
c.execute('UPDATE visits SET count = count + 1')
conn.commit()

# Retrieve updated visitor count
c.execute('SELECT count FROM visits')
visitor_count = c.fetchone()[0]
conn.close()

# Set page config
st.set_page_config(page_title="Renewable Energy Integration Challenges", layout="wide")

# Load images
image_paths = {
    "bhel": "D:/BHEL/Images/bhel.jpg",
    "solar": "D:/BHEL/Images/solar.jpg",
    "wind": "D:/BHEL/Images/wind_power.jpg",
    "hybrid": "D:/BHEL/Images/hybird.jpg"
}

# Sidebar with BHEL image and visitor counter
if os.path.exists(image_paths["bhel"]):
    st.sidebar.image(image_paths["bhel"], caption="BHEL Vision", use_container_width=True)
else:
    st.sidebar.error("Image not found: BHEL logo")

st.sidebar.markdown("## 🌍 Visitor Counter")
st.sidebar.markdown(f"**Total Visitors 👥 :** {visitor_count}")
st.sidebar.progress(min(visitor_count / 100, 1.0))  # Progress bar effect

# Title and Header with BHEL image on the right
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Renewable Energy Integration Challenges")
with col2:
    if os.path.exists(image_paths["bhel"]):
        st.image(image_paths["bhel"], caption="BHEL Vision", use_container_width=True)
    else:
        st.error("Image not found: BHEL logo")

# Image Row
col1, col2, col3 = st.columns(3)
for col, img_key, caption in zip([col1, col2, col3], ["solar", "wind", "hybrid"],
                                 ["Solar Power Integration", "Wind Power Challenges", "Hybrid Energy Solutions"]):
    if os.path.exists(image_paths[img_key]):
        col.image(image_paths[img_key], caption=caption, use_container_width=True)
    else:
        col.error(f"Image not found: {caption}")

# About Section
st.markdown(
    "### Why This Matters?"
    "\nBHEL is at the forefront of the renewable energy revolution, pioneering advancements in solar, wind, and hybrid energy systems."
    "\nHowever, integrating these sources into the grid presents major challenges, including fluctuations in power supply, stability issues, and storage limitations."
    "\nTo ensure a seamless transition to green energy, cutting-edge technologies must be employed to optimize generation, storage, and distribution."
    "\nBy addressing these challenges, BHEL is shaping a future where sustainability and efficiency go hand in hand."
)

# AI in Renewable Energy
st.markdown("## 🔥 AI-Powered Renewable Energy Optimization")
st.markdown(
    "Artificial Intelligence is transforming the renewable energy sector by enhancing forecasting, grid stability, and energy management."
    "\nBy analyzing real-time weather data, consumption patterns, and market demands, AI-driven models can accurately predict power generation and optimize distribution."
    "\nThese intelligent systems reduce energy wastage, prevent blackouts, and ensure a reliable and cost-effective power supply."
    "\nFrom automating grid adjustments to improving battery storage efficiency, AI is a game-changer in making renewable energy more accessible and sustainable for all."
)

# Call to Action
st.markdown(
    "### 🚀 Join the Revolution!"
    "\nHelp us integrate AI with renewable energy for a sustainable future."
    "\n👉 [Learn More](https://www.bhel.com/)"
)