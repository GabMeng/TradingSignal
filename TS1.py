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

class TS:
    def __init__(self, dataset):
        self.dataset = dataset

    def buyandhold(self, column, tickers, initial_value):
        final_values = {}
        for ticker in tickers:
            if ticker in self.dataset:  
                final_value = initial_value
                for x in self.dataset[ticker][column]:
                    final_value *= np.exp(x)
                final_values[ticker] = final_value
        return final_values



    

    def BollingerB(self, tickers, column, ma, mstd, nstd):
        for ticker in tickers:
            if ticker in self.dataset:
                self.dataset[ticker]["MA"] = self.dataset[ticker][column].rolling(window=ma).mean().bfill()
                self.dataset[ticker]["MSTD"] = self.dataset[ticker][column].rolling(window=mstd).std().bfill()
                self.dataset[ticker]["Upper Band"] = self.dataset[ticker]["MA"] + nstd*self.dataset[ticker]["MSTD"]
                self.dataset[ticker]["Lower Band"] = self.dataset[ticker]["MA"] - nstd*self.dataset[ticker]["MSTD"]

                plt.figure(figsize=(12,6))
                plt.plot(self.dataset[ticker]['Adj Close'], label='Adj Close')
                plt.plot(self.dataset[ticker]['Upper Band'], label='Upper Band', color='green')
                plt.plot(self.dataset[ticker]['Lower Band'], label='Lower Band', color='red')
                plt.plot(self.dataset[ticker]['MA'], label='Moving Average', color='orange', linestyle='--') 
                plt.title(f'Bollinger Bands for {ticker}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.show()


    def RSI(self, column, tickers, n=14):
        for ticker in tickers:
            if ticker in self.dataset:
                delta = self.dataset[ticker][column].diff().dropna()
                up, down = delta.copy(), delta.copy()
                up[up < 0] = 0
                down[down > 0] = 0
    
                roll_up = up.rolling(window=n).mean()
                roll_down = down.abs().rolling(window=n).mean()
    
                RS = roll_up / roll_down
                RSI = 100 - (100 / (1.0 + RS))
    
                self.dataset[ticker]['RSI'] = RSI
                plt.figure(figsize=(50,6))
                plt.plot(self.dataset[ticker]['RSI'], label='RSI')
                plt.title(f'RSI for {ticker}')
                plt.axhline(100, linestyle='--', color='red')
                plt.axhline(0, linestyle='--', color='green')
                plt.xlabel('Date')
                plt.ylabel('RSI')
                plt.legend()
                plt.show()


    def BBRSI(self, tickers, column):
        for ticker in tickers:
            if ticker in self.dataset:
                if "Buy Signal" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Buy Signal"])
                if "Sell Signal" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Sell Signal"])
                self.dataset[ticker]["Buy Signal"] = (self.dataset[ticker]["Adj Close"] < self.dataset[ticker]["Lower Band"]) & (self.dataset[ticker]["RSI"] < 20)
                self.dataset[ticker]["Sell Signal"] = (self.dataset[ticker]["Adj Close"] > self.dataset[ticker]["Upper Band"]) & (self.dataset[ticker]["RSI"] > 85)
                
                plt.figure(figsize=(12,6))
                plt.plot(self.dataset[ticker]['Adj Close'], label='Adj Close', alpha=0.5)
                plt.plot(self.dataset[ticker]['Upper Band'], label='Upper Band', color='green', linestyle='--')
                plt.plot(self.dataset[ticker]['Lower Band'], label='Lower Band', color='red', linestyle='--')
                plt.plot(self.dataset[ticker]['MA'], label='Moving Average', color='orange', linestyle='--')
                plt.scatter(self.dataset[ticker][self.dataset[ticker]["Buy Signal"]].index, self.dataset[ticker][self.dataset[ticker]["Buy Signal"]]["Adj Close"], color='green', marker='^', alpha=1)
                plt.scatter(self.dataset[ticker][self.dataset[ticker]["Sell Signal"]].index, self.dataset[ticker][self.dataset[ticker]["Sell Signal"]]["Adj Close"], color='red', marker='v', alpha=1)
                plt.title(f'Bollinger Bands for {ticker}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.show()


    def trendStalker(self, tickers, column):
        for ticker in tickers:
            if ticker in self.dataset:
                if "Buy Signal" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Buy Signal"])
                if "Sell Signal" in self.dataset[ticker]:
                    self.dataset[ticker] = self.dataset[ticker].drop(columns=["Sell Signal"])
                self.dataset[ticker]["MACD"] = self.dataset[ticker][column].ewm(span=12, adjust=False).mean() - self.dataset[ticker][column].ewm(span=26, adjust=False).mean()
                self.dataset[ticker]["Signal Line"] = self.dataset[ticker]["MACD"].ewm(span=9, adjust=False).mean()
                self.dataset[ticker]["Buy Signal"] = self.dataset[ticker]["MACD"] > self.dataset[ticker]["Signal Line"]
                self.dataset[ticker]["Sell Signal"] = self.dataset[ticker]["MACD"] < self.dataset[ticker]["Signal Line"]
                plt.figure(figsize=(80,6))
                plt.plot(self.dataset[ticker]["MACD"], label='MACD', color='blue')
                plt.plot(self.dataset[ticker]["Signal Line"], label='Signal Line', color='red')
                plt.title(f'MACD for {ticker}')
                plt.legend()
                plt.show()



    

    def backtestsimple(self, tickers):
        portfolio = {}
        for ticker in tickers:
            portfolio[ticker] = {'position': 'out', 'cash': 1000000, 'shares': 0}

        for ticker in tickers:
            if ticker in self.dataset:
                for i in range(len(self.dataset[ticker])):
                    if self.dataset[ticker]["Buy Signal"].iloc[i]:
                        if portfolio[ticker]['position'] == 'out':
                            portfolio[ticker]['shares'] = portfolio[ticker]['cash'] / self.dataset[ticker]['Adj Close'].iloc[i]
                            portfolio[ticker]['cash'] = 0
                            portfolio[ticker]['position'] = 'long'
                    elif self.dataset[ticker]["Sell Signal"].iloc[i]:
                        if portfolio[ticker]['position'] == 'long':
                            portfolio[ticker]['cash'] = portfolio[ticker]['shares'] * self.dataset[ticker]['Adj Close'].iloc[i]
                            portfolio[ticker]['shares'] = 0
                            portfolio[ticker]['position'] = 'out'

        total_value = sum(portfolio[ticker]['cash'] + portfolio[ticker]['shares'] * self.dataset[ticker]['Adj Close'].iloc[-1] for ticker in tickers)
        total_portfolio_return = round(((total_value - 1000000) / 1000000)*100, 3)
        annualized_portfolio_return = round(((1 + total_portfolio_return/100)**(252/len(self.dataset[tickers[0]])) - 1)*100, 3)
        return f"Final Value:{total_value}, Total Return:{total_portfolio_return}%, Annualized Return:{annualized_portfolio_return}%"




        






            
        



        


    
   
    








   