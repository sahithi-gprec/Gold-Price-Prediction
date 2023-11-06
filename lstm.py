import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

# Load the dataset
df = pd.read_csv('Gold Price.csv')
df.drop(['Volume', 'Chg%'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")  # Corrected date format
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Create the Streamlit app
st.title("Gold Price Prediction App")

st.sidebar.header("Settings")
test_size = st.sidebar.number_input("Test Set Size (year 2022)", value=200, min_value=1)
window_size = st.sidebar.number_input("Window Size", value=60, min_value=1)

# Show the gold price history chart
st.header("Gold Price History Data")
fig = px.line(y=df.Price, x=df.Date)
fig.update_traces(line_color='black') 
fig.update_layout(xaxis_title="Date", yaxis_title="Scaled Price", title={'text': "Gold Price History Data"})
st.plotly_chart(fig)

# Split the data and build the model
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

X_train, y_train, X_test, y_test = [], [], [], []

# Data preprocessing
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

test_data = df.Price[-test_size - window_size:]
test_data = scaler.transform(test_data.values.reshape(-1, 1))

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    y_test.append(test_data[i, 0])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# Model definition
def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()

    return model

model = define_model()

# Model training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
result = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE

# Display model performance
st.header("Model Performance on Gold Price Prediction")

y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)

plt.figure(figsize=(15, 6), dpi=150)
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('axes', edgecolor='grey')
plt.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='blue', lw=2)
plt.plot(df['Date'].iloc[-test_size:], y_test_true, color='red', lw=2)
plt.plot(df['Date'].iloc[-test_size:], y_test_pred, color='green', lw=2)
plt.title('Model Performance on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})

# Streamlit chart to display the performance plot
st.pyplot(plt)
