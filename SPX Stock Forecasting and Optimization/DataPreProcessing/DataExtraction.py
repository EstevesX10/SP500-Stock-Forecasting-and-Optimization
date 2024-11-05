import pandas as pd
import os
import yfinance as yf
from datetime import (datetime)

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

def strToDatetime(string_date:str) -> datetime:
    """
    # Description
        -> Converts a string of type YYYY-MM-DD into a datetime object. 
    -------------------------------------------------------------------
    := param: string_date - String that we want to convert into a datetime type object [Eg: '2003-10-10'].
    := return: Instance of Datetime based on the given date string.
    """
    # Fetching the year, month and day from the string and convert them into int
    year, month, day = list(map(int, string_date.split('-')))

    # Return a instance of datetime with the respective extracted attributes from the given string
    return datetime(year=year, month=month, day=day)

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
        stockHistory['Date'] = stockHistory['Date'].apply(lambda x: str(x).split(' ')[0])
        stockHistory['Date'] = stockHistory['Date'].map(lambda dateString : strToDatetime(dateString))

        # Replace the index with the 'Date'
        stockHistory.index = stockHistory['Date']

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
    stockFilePath = pathsConfig['Datasets']['Stocks-Market-Information'] + f"/{stockSymbol}.csv"

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
        stockHistory['Date'] = stockHistory['Date'].apply(lambda x: str(x).split(' ')[0])
        stockHistory['Date'] = stockHistory['Date'].map(lambda dateString : strToDatetime(dateString))

        # Replace the index with the 'Date'
        stockHistory.index = stockHistory['Date']

        # Saving the History data into a csv file
        stockHistory.to_csv(stockFilePath, sep=',', index=False)

    else:
        # Read the previously computed data into a DataFrame
        stockHistory = pd.read_csv(stockFilePath)

    # Return the stock history
    return stockHistory