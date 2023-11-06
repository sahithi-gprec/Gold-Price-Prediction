import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load your data from 'Gold Price.csv'
data = pd.read_csv('Gold Price.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

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

# Split the data into training and testing sets (for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Input for user date
user_date = input("Enter a date (YYYY-MM-DD): ")
user_date = pd.to_datetime(user_date)

# Extract date-related features from user input
user_day_of_year = user_date.dayofyear
user_month = user_date.month
user_day_of_week = user_date.dayofweek

# Input from the user for 'Open' and 'Volume' features
user_open = float(input("Enter the 'Open' value: "))
user_volume = float(input("Enter the 'Volume' value: "))

# Create a feature vector for prediction with user-provided values
user_input = [user_day_of_year, user_month, user_day_of_week, user_open, user_open, user_open, user_volume]

# Make predictions based on user-provided date and features
predicted_gold_price = model.predict([user_input])

# Calculate regression metrics on the testing data
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Predict 'user_high', 'user_low', and 'user_chg_percent' values using the model
predicted_high = predicted_gold_price + user_open
predicted_low = predicted_gold_price - user_open
predicted_chg_percent = (predicted_gold_price - user_open) / user_open

print(f'Predicted gold price for the user-provided date: {predicted_gold_price[0]:.2f}')
print(f'Predicted change percentage: {predicted_chg_percent[0] * 100:.6f}%')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R^2) Score: {r2:.2f}')
