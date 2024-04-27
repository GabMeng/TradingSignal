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

class DataAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset

    def getADF(self, Tick, Column):
        for ticker in Tick:
            if ticker in self.dataset:
                result = adfuller(self.dataset[ticker][Column])
                print(f'ADF Statistic for {ticker} {Column}: {result[0]}')
                print(f'p-value for {ticker} {Column}: {result[1]}')
                if result[1] < 0.05:
                    print('\033[92m'+f"{ticker} {Column} is stationary")
                else: 
                    print('\033[91m'+f"{ticker} {Column} is not stationary")   
            else: 
                print(f"No data for {ticker}")

    def plotDistr(self, Tick, Column):
        for ticker in Tick:
            if ticker in self.dataset:
                sns.displot(self.dataset[ticker][Column], kde=True)
                plt.show()
            else:
                print(f"No data for {ticker}")

    def getQQPlotted(self, Tick, Column):
        for ticker in Tick:
            if ticker in self.dataset:
                fig = sm.qqplot(self.dataset[ticker][Column], line='45')
                plt.show()

    def getSWTested(self, Tick, Column):    
        for ticker in Tick:
            if ticker in self.dataset:
                result = stats.shapiro(self.dataset[ticker][Column])
                print(f"Shapiro-Wilk test for {ticker} {Column}: {result}")
                if result[1] > 0.05:
                    print('\033[92m'+"Data is normally distributed according to Shapiro-Wilk test"+'\033[0m')
                else:
                    print('\033[91m'+"Data is not normally distributed according to Shapiro-Wilk test"+'\033[0m')
            else:
                print(f"Ticker {ticker} not found in the dataset")
                
    def getKSTested(self, Tick, Column):
        for ticker in Tick:
            if ticker in self.dataset:
                for col in Column:
                    result = stats.kstest(self.dataset[ticker][col], 'norm')
                    print(f"Kolmogorov-Smirnov test for {ticker} {col}: {result}")
                    if result[1] > 0.05:
                        print('\033[92m'+"Data is normally distributed"'\033[0m')
                    else:
                        print('\033[91m'+"Data is not normally distributed according to Kolmogorov-Smirnov test"'\033[0m')
            else:
                print(f"Ticker {ticker} not found in the dataset")
               



  

