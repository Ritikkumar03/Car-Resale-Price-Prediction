import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# Load data and model
df = pd.read_csv('clean_df.csv')  # Updated CSV file name
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title="Car Resale Price Prediction", layout="wide")

# Apply custom dark mode style with button hover effect
st.markdown(
    """
    <style>
        body {
            color: #e0e0e0;
            background-color: #ddd8d5;
        }
        .stButton > button {
            background-color: #EFC070;
            color: white;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #EFC070; /* Warm light color */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("Car Resale Price Prediction")
st.markdown("An interactive Web app to predict and analyze car resale prices with charts.")

# Sidebar filters
st.sidebar.header("Filter Options")
brand = st.sidebar.selectbox("Select Car Brand", options=[None] + list(df['brand'].unique()))
car_name = st.sidebar.selectbox(
    "Select Car Model", options=[None] + (list(df[df['brand'] == brand]['full_name'].unique()) if brand else [])
)
year = st.sidebar.selectbox("Select Registered Year", options=[None] + list(sorted(df['registered_year'].unique(), reverse=True)))
transmission = st.sidebar.selectbox("Select Transmission Type", options=[None] + list(df['transmission_type'].unique()))
fuel = st.sidebar.selectbox("Select Fuel Type", options=[None] + list(df['fuel_type'].unique()))
owner = st.sidebar.selectbox("Select Owner Type", options=[None] + list(df['owner_type'].unique()))
insurance = st.sidebar.selectbox("Select Insurance Type", options=[None] + list(df['insurance'].unique()))
kms_driven = st.sidebar.slider("Kilometers Driven", min_value=500, max_value=150000, step=500)

# Calculate car age if year is selected
age = None if year is None else 2023 - year

# Predict resale price button (just below Kilometers Driven slider)
st.sidebar.markdown("### Predict Resale Price")
if st.sidebar.button("Predict Resale Price"):
    if car_name:
        car_engine_capacity = df[df['full_name'] == car_name]['engine_capacity'].iloc[0]
        result = pipe.predict([[car_name, car_engine_capacity, year, insurance, transmission, kms_driven, owner, fuel, brand, age]])
        result = np.round(np.exp(result[0]))
        st.sidebar.success(f"Predicted Resale Price: **₹{result}**")
    else:
        st.sidebar.warning("Please select a car model.")

# Filter data based on selected car model
filtered_df = df[df['full_name'] == car_name] if car_name else df

# Add charts
st.write("### Data Analysis")
col1, col2 = st.columns(2)

# Chart 1: Price distribution by brand
with col1:
    if car_name:
        fig1 = px.box(
            filtered_df, x="brand", y="resale_price", color="brand",
            title=f"Resale Price Distribution for {car_name}",
            labels={"resale_price": "Resale Price (₹)", "brand": "Car Brand"},
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.write("Select a car model to view this chart.")

# Chart 2: Resale price vs. kilometers driven
with col2:
    if car_name:
        fig2 = px.scatter(
            filtered_df, x="kms_driven", y="resale_price", color="fuel_type",
            title=f"Resale Price vs. Kilometers Driven for {car_name}",
            labels={"kms_driven": "Kilometers Driven", "resale_price": "Resale Price (₹)"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("Select a car model to view this chart.")

# Chart 3: Average resale price by year
if car_name:
    st.write("### Average Resale Price by Registered Year")
    avg_price_year = filtered_df.groupby("registered_year")["resale_price"].mean().reset_index()
    fig3 = px.line(
        avg_price_year, x="registered_year", y="resale_price",
        title=f"Average Resale Price Over the Years for {car_name}",
        labels={"registered_year": "Registered Year", "resale_price": "Average Resale Price (₹)"},
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("Select a car model to view the average resale price chart.")

# Footer
st.markdown("---")
st.markdown(
    "Developed by Ritik Kumar. For queries, contact: ritikdce@gmail.com.",
    unsafe_allow_html=True,
)
