import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load your data from 'Gold Price.csv'
data = pd.read_csv('Gold Price.csv')

# Convert the 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Extract date-related features
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Select the features (independent variables) and the target variable (Gold_Price)
X = data[['DayOfYear', 'Month', 'DayOfWeek', 'Open', 'High', 'Low', 'Volume']]
y = data['Price']

# Create a linear regression model
model = LinearRegression()

# Train the model on the entire dataset
model.fit(X, y)

# Streamlit app
st.title("Gold Price Prediction App")

# Input for user date
user_date = st.date_input("Enter a date:")

# Extract date-related features from user input
user_day_of_year = user_date.timetuple().tm_yday
user_month = user_date.month
user_day_of_week = user_date.weekday()

# Input from the user for 'Open' and 'Volume' features
user_open = st.number_input("Enter the 'Open' value:", value=0.0)
user_volume = st.number_input("Enter the 'Volume' value:", value=0.0)

# Create a feature vector for prediction with user-provided values
user_input = [user_day_of_year, user_month, user_day_of_week, user_open, user_open, user_open, user_volume]

# Button to trigger prediction
if st.button("Predict the Price"):
    # Make predictions based on user-provided date and features
    predicted_gold_price = model.predict([user_input])
    
    # Predict 'user_high', 'user_low', and 'user_chg_percent' values using the model
    # predicted_high = predicted_gold_price + user_open
    # predicted_low = predicted_gold_price - user_open
    predicted_chg_percent = ((predicted_gold_price - user_open) / user_open) * 100
    
    st.write(f'Predicted gold price for the user-provided date: ${predicted_gold_price[0]:.2f}')
    # st.write(f'Predicted high: ${predicted_high[0]:.2f}')
    # st.write(f'Predicted low: ${predicted_low[0]:.2f}')
    st.write(f'Predicted change percentage: {predicted_chg_percent.item():.6f}%')
