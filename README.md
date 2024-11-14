# StockMarketPrediction
# Stock Market Prediction Using TBATS Model

This repository contains a Python implementation for **stock market prediction** using the **TBATS (Trend and Seasonal BATS)** model. TBATS is an extension of the **BATS (Box-Cox ARMA Trend Seasonal)** model, which is specifically designed to handle complex seasonality and non-linear trends, making it suitable for forecasting time-series data like stock market prices.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Implementation](#model-implementation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project applies the **TBATS model** for predicting stock prices based on historical data. The TBATS model is robust in handling multiple seasonal patterns and non-linear trends, which are common in stock market data. This implementation leverages **Python** and uses popular libraries like **statsmodels**, **pandas**, and **tbats** for time-series forecasting.

The aim of this project is to demonstrate how the TBATS model can be used to predict stock prices with high accuracy and evaluate its performance in comparison to traditional models like ARIMA and Exponential Smoothing.

## Features

- **Stock Data Fetching**: Fetches historical stock data from online APIs like **Yahoo Finance**.
- **Data Preprocessing**: Includes handling missing values, normalization, and feature engineering.
- **TBATS Forecasting**: Implements TBATS model for trend and seasonality detection.
- **Model Comparison**: Compares TBATS with traditional models like **ARIMA** and **ETS**.
- **Performance Metrics**: Evaluates model performance using metrics like **MAE**, **RMSE**, and **MAPE**.
- **Visualization**: Provides graphical visualizations for stock prices, predicted values, and residuals.

## Requirements

- Python 3.6+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `statsmodels`
  - `tbats`
  - `yfinance`
  - `sklearn`
  - `seaborn`
  
You can install these libraries using `pip`:

```bash
pip install numpy pandas matplotlib statsmodels tbats yfinance sklearn seaborn
Installation
To get started with this project, follow the steps below to clone the repository and install the necessary dependencies.

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/stock-market-tbats.git
cd stock-market-tbats
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Fetching Data: The stock data is fetched using the yfinance package, which allows downloading historical stock data for any stock ticker. The data is loaded and processed for the TBATS model.
python
Copy code
import yfinance as yf

# Example: Fetch stock data for Apple (AAPL)
data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
Data Preprocessing: Before applying the TBATS model, data is preprocessed to handle missing values and normalize prices. The data is then split into training and testing sets.
python
Copy code
# Preprocess data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
Model Implementation: The TBATS model is applied to the stock data using the tbats library.
python
Copy code
from tbats import TBATS

# Fit TBATS model
model = TBATS(seasonal_periods=(5, 7, 365))  # Example for multiple seasonalities
fitted_model = model.fit(scaled_data)

# Make predictions
forecast = fitted_model.forecast(steps=30)  # Predict next 30 days
Model Evaluation: Evaluate the model using metrics like MAE, RMSE, and MAPE.
python
Copy code
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate MAE, RMSE, and MAPE
mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
Visualization: Visualize the stock prices, forecasted values, and residuals using matplotlib.
python
Copy code
import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted')
plt.legend()
plt.show()
Model Implementation
The TBATS model used in this project is an extension of the BATS (Box-Cox ARMA Trend Seasonal) model. TBATS can handle multiple seasonalities (e.g., daily, weekly, yearly cycles), non-linear trends, and the inherent volatility present in financial data.

Model Parameters:
Trend: Captures the long-term movement of stock prices.
Seasonality: Identifies multiple seasonal patterns, such as weekly or yearly cycles.
ARMA: Models the residual errors from the trend and seasonality components.
Box-Cox Transformation: Stabilizes variance across the series to improve prediction accuracy.
Results
This section compares the performance of the TBATS model with other forecasting models like ARIMA and ETS. Evaluation metrics such as MAE, RMSE, and MAPE will be provided to demonstrate the model's accuracy.

Visualizations of forecasted prices, as well as performance comparisons, will be included in this section.

Contributing
If you'd like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request. Contributions are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.
