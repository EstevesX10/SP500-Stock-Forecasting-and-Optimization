import numpy as np
import pandas as pd
import os
from pathlib import (Path)
import yfinance as yf
from datetime import datetime as dt

def extractSP500StocksInformationWikipedia(pathsConfig:dict=None) -> pd.DataFrame:
    """
    # Description
        -> This function helps extract some information regarding the S&P-500 Stock Options [From Wikipedia].
    ---------------------------------------------------------------------------------------------------------
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: Pandas Dataframe with the information collected.
    """

    # Check if the pathsConfig was passed on
    if pathsConfig is None:
        raise ValueError("Missing a Paths Configuration Dictionary!")
    
    # Check if the information was previously computed
    if not os.path.exists(pathsConfig['Datasets']['SP500-Stocks-Wikipedia']):
        # URL of the Wikipedia page containing the list of S&P 500 companies
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # Read the HTML tables from the page
        tables = pd.read_html(url)
        
        # The first table on the page contains the list of S&P 500 companies
        sp500Stocks = tables[0].sort_values(by='Symbol')
        
        # Reset the indices
        sp500Stocks = sp500Stocks.reset_index().drop('index', axis=1)

        # Save the DataFrane
        sp500Stocks.to_csv(pathsConfig['Datasets']['SP500-Stocks-Wikipedia'], sep=',', index=False)

    else:
        # Load the data
        sp500Stocks = pd.read_csv(pathsConfig['Datasets']['SP500-Stocks-Wikipedia'])

    # Return the DataFrame
    return sp500Stocks

def getSP500StockMarketInformation(config:dict=None, pathsConfig:dict=None) -> pd.DataFrame:
    """
    # Description
        -> This function helps extract the Market Information of the S&P-500 Stock.
    -------------------------------------------------------------------------------
    := param: config - Dictionary with constants used to define the interval in which to extract the stock's information from.
    := param: pathsConfig - Dictionary used to manage file paths. 
    := return: Pandas DataFrame with the extracted market information.
    """

    # Verify if the config was given
    if config is None:
        raise ValueError("Missing a Configuration Dictionary!")

    # Check if the pathsConfig was also passed on
    if pathsConfig is None:
        raise ValueError("Missing a Paths Configuration Dictionary!")
    
    # Define the stock symbol for the S&P-500
    stockSymbol = '^GSPC'

    # Define the file path in which the stock's market information resides in
    stockFilePath = pathsConfig['Datasets']['SP500-Market-Information']

    # Check if the information has already been fetched
    if not os.path.exists(stockFilePath):
        try:
            # Getting the Stock Market Information
            stockInformation = yf.Ticker(stockSymbol)
        except:
            # The stock is not available through the yahoo finance API
            print(f"[{stockSymbol}] Invalid Stock!")
            return None
        
        # Fetching a dataset with the stock's history data
        if config['max_period']:
            stockHistory = stockInformation.history(period="max")
        else:
            stockHistory = stockInformation.history(start=config['start_date'], end=config['end_date'])

        # Get the index back into the DataFrame
        stockHistory = stockHistory.reset_index()

        # Adapt the Date on the dataframe to simply include the date and not the time
        stockHistory['Date'] = pd.to_datetime(stockHistory['Date']).dt.date

        # Create a Daily Return with help of the pct_change
        stockHistory['Daily Return'] = stockHistory['Close'].pct_change()

        # Replace the index with the 'Date'
        stockHistory.index = stockHistory['Date']

        # Configure dataframe to be placed only beyond 2010
        stockHistory = stockHistory[stockHistory['Date'] > dt(2010, 1, 1).date()]

        # Get the index back into the DataFrame
        stockHistory = stockHistory.reset_index(drop=True)

        # Saving the History data into a csv file
        stockHistory.to_csv(stockFilePath, sep=',', index=False)

    else:
        # Read the previously computed data into a DataFrame
        stockHistory = pd.read_csv(stockFilePath)

    # Return the stock history
    return stockHistory

def getStockMarketInformation(stockSymbol:str=None, config:dict=None, pathsConfig:dict=None) -> pd.DataFrame:
    """
    # Description
        -> This function helps extract the Market Information of a given Stock.
    ---------------------------------------------------------------------------
    := param: stockSymbol - Stock that we aim to extract.
    := param: config - Dictionary with constants used to define the interval in which to extract the stock's information from.
    := param: pathsConfig - Dictionary used to manage file paths. 
    := return: Pandas DataFrame with the extracted market information.
    """

    # Check if the stock was passed on
    if stockSymbol is None:
        raise ValueError("Missing a Stock to extract the data from!")

    # Verify if the config was given
    if config is None:
        raise ValueError("Missing a Configuration Dictionary!")

    # Check if the pathsConfig was also passed on
    if pathsConfig is None:
        raise ValueError("Missing a Paths Configuration Dictionary!")
    
    # Define the file path in which the stock's market information resides in
    stockFilePath = pathsConfig['Datasets']['Raw-Stocks-Market-Information'][f"{stockSymbol}"]

    # Define the Folder in which to save the windowed DataFrame
    stockFileFolder = Path("/".join(stockFilePath.split("/")[:-1]))
    
    # Check if the directory exists. If not create it
    stockFileFolder.mkdir(parents=True, exist_ok=True)

    # Check if the information has already been fetched
    if not os.path.exists(stockFilePath):
        try:
            # Getting the Stock Market Information
            stockInformation = yf.Ticker(stockSymbol)
        except:
            # The stock is not available through the yahoo finance API
            print(f"[{stockSymbol}] Invalid Stock!")
            return None
        
        # Fetching a dataset with the stock's history data
        if config['max_period']:
            stockHistory = stockInformation.history(period="max")
        else:
            # Fetch the whole data
            data = stockInformation.history(period="max")
            
            # Old Stock and therefore it has been delisted
            if data.shape[0] == 0:
                print(f"[{stockSymbol}] Delisted Stock")
                return None
            
            # Get the earliest available date
            startDate = max(data.index.min().strftime('%Y-%m-%d'), config['start_date'])

            # print("Start Date", startDate)
            # print("End Date", config['end_date'])

            # Recent DataFrames that go into 2024
            if (startDate > config['end_date']):
                print(f"[{stockSymbol}] Added too recently to be used!")
                return None

            # Grab the data for the computed interval
            stockHistory = stockInformation.history(start=startDate, end=config['end_date'])
            
        # Get the index back into the DataFrame
        stockHistory = stockHistory.reset_index()

        # Adapt the Date on the dataframe to simply include the date and not the time
        stockHistory['Date'] = pd.to_datetime(stockHistory['Date']).dt.date

        # Calculate the Simple Moving Average - Calculates the N-Day SMA for closing prices, 
        # providing a view of the stock's trend
        stockHistory['SMA'] = stockHistory['Close'].rolling(window=config['window']).mean()

        # Calculate the Exponential Moving Average
        # It gives more weight to recent prices, offering a closer look at the current trend
        stockHistory['EMA'] = stockHistory['Close'].ewm(span=config['window'], adjust=False).mean()

        # Calculate the Bollinger Bands used to assess volatility and potential overbought/oversold stocks
        stockHistory['UpperBB'] = stockHistory['SMA'] + (stockHistory['Close'].rolling(window=config['window']).std() * 2)
        stockHistory['LowerBB'] = stockHistory['SMA'] - (stockHistory['Close'].rolling(window=config['window']).std() * 2)

        # Create a Daily Return with help of the pct_change
        stockHistory['Daily_Return'] = stockHistory['Close'].pct_change()

        # Calculate the cumulative return
        stockHistory['Cumulative_Return'] = (1 + stockHistory['Daily_Return']).cumprod()

        # Define the Window Return
        stockHistory['Window_Return'] = stockHistory['Close'].pct_change(periods=config['window'])

        # Compute the Volatility based on the window return
        stockHistory['Volatility'] = stockHistory['Window_Return'].rolling(window=config['window']).std()

        # Configure dataframe to be placed only beyond 2010 and before February 2024
        stockHistory = stockHistory[stockHistory['Date'] >= dt(2010, 1, 1).date()]
        stockHistory = stockHistory[stockHistory['Date'] < dt(2024, 2, 1).date()]

        # Get the index back into the DataFrame
        stockHistory = stockHistory.reset_index(drop=True)

        # Check for mismatches based on the volatility window, since some metrics are computed based on the previous N Entries
        if config['start_date'] < startDate:
            stockHistory = stockHistory.iloc[config['window']:]

        # Saving the History data into a csv file
        stockHistory.to_csv(stockFilePath, sep=',', index=False)

    else:
        # Read the previously computed data into a DataFrame
        stockHistory = pd.read_csv(stockFilePath)

    # Return the stock history
    return stockHistory