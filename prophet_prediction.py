import streamlit as st
import pandas as pd
from prophet import Prophet

# Load and preprocess the historical data
df = pd.read_csv('gold_price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = df['Price'].str.replace(',', '').astype(float)  # Convert 'Price' to float

# Rename the columns to fit Prophet's requirements
df = df.rename(columns={'Date': 'ds', 'Price': 'y'})

# Create and fit the Prophet model
model = Prophet()
model.fit(df)

# Streamlit web app
st.title("Gold Price Prediction App")

# Input: User-provided future date
user_date = st.date_input("Enter a future date for prediction (e.g., 2023-11-07):")

if user_date:
    # Create a dataframe with the user-provided date
    future = pd.DataFrame({'ds': [pd.to_datetime(user_date)]})

    # Make predictions for the user-provided date
    forecast = model.predict(future)

    # Display predicted price for the user-provided date
    predicted_price = forecast['yhat'].values[0]
    st.write(f'Predicted Price for {user_date}: Rs. {predicted_price:.2f}')
