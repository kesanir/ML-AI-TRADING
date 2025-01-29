#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf


# In[29]:


# For fetching financial data  
class DataFetcher:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    def fetch_data(self):
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            return data


# In[32]:


class FeatureEngineer:
    def __init__(self, data):
        self.data = data
    def create_features(self):
            # Example features 
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['RSI'] = self.calculate_rsi()
            return self.data.dropna()
    def calculate_rsi(self, period=14):
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))


# In[36]:


class Model:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
    def train(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            return mse
    def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)


# In[41]:


class Strategy:
    def __init__(self, model, threshold=0.005):
        self.model = model
        self.threshold = threshold
    def make_decision(self, features):
        prediction = self.model.predict(features.reshape(1, -1))
        if prediction > self.threshold:
            return "Buy"
        elif prediction < -self.threshold:
             return "Sell"
        else:
             return prediction


# In[27]:


class HedgeFundTeam:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.data_fetcher = DataFetcher(ticker, start_date, end_date)
        self.feature_engineer = None
        self.model = Model()
        self.strategy = None
    def run(self):
            data = self.data_fetcher.fetch_data()
            self.feature_engineer = FeatureEngineer(data)
            features = self.feature_engineer.create_features()
            X = features[['SMA_20', 'RSI']].values
            y = features['Close'].pct_change().shift(-1).fillna(0).values
            mse = self.model.train(X, y)
            print(f"Model Mean Squared Error: {mse}")
            self.strategy = Strategy(self.model)
            # Simulate trading decisions for the last few days 
            for i in range(-10, 0):
                # Last 10 days
                decision = self.strategy.make_decision(X[i])
                print(f"Day {i}: Decision - {decision}")          


# In[46]:


# Usage 
team = HedgeFundTeam('AAPL', '2021-01-01', '2025-01-27')
team.run()


# In[47]:


# Usage 
team = HedgeFundTeam('TSLA', '2021-01-01', '2025-01-27')
team.run()


# In[48]:


# Usage 
team = HedgeFundTeam('^GSPC', '2021-01-01', '2025-01-27')
team.run()


# In[50]:


SP500 = yf.download('^GSPC','2021-01-01', '2025-01-27')


# In[51]:


import matplotlib.pyplot as plt

SP500['Close'].plot(figsize=(10, 6))
plt.title('S&P 500 Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.show()


# In[52]:


# Save the data to a CSV file
SP500.to_csv('sp500_data.csv')


# In[ ]:




