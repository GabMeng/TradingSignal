import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import yfinance as yf   
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro 
import math
from scipy import stats
from scipy.stats import kstest
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

class ML:
    def __init__(self, dataset):
        self.dataset = dataset






    def linearreg(self, tickers, column, n):
        for ticker in tickers:
            if ticker in self.dataset:
                if "Prediction" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Prediction"])
                self.dataset[ticker]["Prediction"] = self.dataset[ticker][column].shift(-n)
                self.dataset[ticker] = self.dataset[ticker].dropna()
                X = self.dataset[ticker][column].values.reshape(-1,1)
                y = self.dataset[ticker]["Prediction"].values.reshape(-1,1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                print(f"Mean Squared Error for {ticker}: {mse}")
                print(f"R^2 for {ticker}: {r2}")
                plt.scatter(X_test, y_test, color='black')
                plt.plot(X_test, predictions, color='blue', linewidth=3)
                plt.title(f'Linear Regression for {ticker}')
                plt.xlabel(column)
                plt.ylabel('Prediction')
                plt.show()



    def NN(self, tickers, column, n):
        for ticker in tickers:
            if ticker in self.dataset:
                if "Prediction" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Prediction"])   
                self.dataset[ticker]["Prediction"] = self.dataset[ticker][column].shift(-n)   
                self.dataset[ticker].dropna(subset=["Prediction"], inplace=True)  
                X = self.dataset[ticker][column].values.reshape(-1,1)
                y = self.dataset[ticker]["Prediction"].values.reshape(-1,1)  
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train.ravel())
                y_pred = model.predict(X_test_scaled)

                mse = mean_squared_error(y_test, y_pred)
                print("Mean Squared Error:", mse)