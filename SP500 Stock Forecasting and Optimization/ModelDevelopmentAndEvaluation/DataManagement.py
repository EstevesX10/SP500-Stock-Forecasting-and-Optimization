from typing import (List)
import pandas as pd
import os

def getStocksPredictions(stocks:List[str]=None, pathsConfig:dict=None) -> pd.DataFrame:
    """
    # Description
        -> This function helps me merge all the predicted closing Prices
        of all the selected stocks in order to properly used them for the Portfolio Optimization.
    ---------------------------------------------------------------------------------------------
    := param: stocks - List with all the stocks to consider.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: DataFrame with all the predicted closing prices for each stock for January 2024.
    """

    # Check if a list of stocks was given
    if stocks is None:
        raise ValueError("Missing a List of Stocks whoose Closing Prices we are to merge into a single DataFrame!")

    # Verify if the stocks list contains any elements
    if len(stocks) == 0:
        raise ValueError("Empty List of Stocks!")

    # Check if the pathsConfig was also passed on
    if pathsConfig is None:
        raise ValueError("Missing a Paths Configuration Dictionary!")

    # Get the path to store the DataFrame with all the stock's closing prices
    stockPredictionsPath = pathsConfig['ExperimentalResults']['Final-Predictions']

    # Check if the DataFrame has already been computed
    if not os.path.exists(stockPredictionsPath):
        # Get first stock
        firstStock = stocks[0]

        # Load the first stock's raw market history dataset
        stocksDataFrame = pd.read_csv(pathsConfig["ExperimentalResults"][firstStock][f"{firstStock}-Predictions"])

        # Iterate through the DataFrames and load them into memory
        for stock in stocks[1:]:
            # Load current stock's market details dataset
            currentStockDataFrame = pd.read_csv(pathsConfig["ExperimentalResults"][stock][f"{stock}-Predictions"])

            # Merge it with the previous DataFrame
            stocksDataFrame = pd.merge(stocksDataFrame, currentStockDataFrame, on='Date', how='outer')

        # Save DataFrame
        stocksDataFrame.to_csv(stockPredictionsPath, sep=',', index=False)

    # Already Computed DataFrame
    else:
        # Load Final DataFrame
        stocksDataFrame = pd.read_csv(stockPredictionsPath)

    # Return the Final DataFrame with the stock's predictions for January 2024
    return stocksDataFrame

def getStocksOpeningPrices(stocks:List[str], pathsConfig:dict) -> pd.DataFrame:
    """
    # Description
        -> This function enables us to collect all the real opening prices from all the stocks.
    -------------------------------------------------------------------------------------------
    := param: stocks - List with the Stocks to consider.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: Pandas DataFrame with all the stock's opening prices.
    """

    # Define the path to store the real opening prices of all the stocks
    openingPricesPath = pathsConfig['ExperimentalResults']['Stocks-Open-Prices']

    # If the DataFrame has yet to be computed
    if not os.path.exists(openingPricesPath):
        # Get the first stock
        firstStock = stocks[0]

        # Initialize the DataFrame for all the data
        openingPrices = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][firstStock])[['Date', 'Open']]

        # Filter only the dates for January 2024
        openingPrices = openingPrices[openingPrices['Date'] >= '2024-01-01'][openingPrices['Date'] <= '2024-01-31']

        # Rename the closing prices columns
        openingPrices = openingPrices.rename(columns={'Open': f'{firstStock}_Open'})

        # Iterate through the available stocks
        for stock in stocks[1:]:
            # Load the next stocks opening prices
            currentStockOpeningPrices = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][stock])[['Date', 'Open']]

            # Filter only the dates for January 2024
            currentStockOpeningPrices = currentStockOpeningPrices[currentStockOpeningPrices['Date'] >= '2024-01-01'][currentStockOpeningPrices['Date'] <= '2024-01-31']

            # Rename the closing prices columns
            currentStockOpeningPrices = currentStockOpeningPrices.rename(columns={'Open': f'{stock}_Open'})

            # Merge the new DataFrame on the initial one
            openingPrices = pd.merge(openingPrices, currentStockOpeningPrices, on='Date', how='outer')

        # Save the Final DataFrame 
        openingPrices.to_csv(openingPricesPath, sep=',', index=False)
    else:
        # Load the Previously computed DataFrame
        openingPrices = pd.read_csv(openingPricesPath)

    # Return the Final DataFrame
    return openingPrices

def getStocksClosingPrices(stocks:List[str], pathsConfig:dict) -> pd.DataFrame:
    """
    # Description
        -> This function enables us to collect all the real closing prices from all the stocks.
    -------------------------------------------------------------------------------------------
    := param: stocks - List with the Stocks to consider.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: Pandas DataFrame with all the stock's opening prices.
    """

    # Define the path to store the real opening prices of all the stocks
    closingPricesPath = pathsConfig['ExperimentalResults']['Stocks-Closing-Prices']

    # If the DataFrame has yet to be computed
    if not os.path.exists(closingPricesPath):
        # Get the first stock
        firstStock = stocks[0]

        # Initialize the DataFrame for all the data
        closingPrices = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][firstStock])[['Date', 'Close']]

        # Filter only the dates for January 2024
        closingPrices = closingPrices[closingPrices['Date'] >= '2024-01-01'][closingPrices['Date'] <= '2024-01-31']

        # Rename the closing prices columns
        closingPrices = closingPrices.rename(columns={'Close': f'{firstStock}_Close'})

        # Iterate through the available stocks
        for stock in stocks[1:]:
            # Load the next stocks closing prices
            currentStockClosingPrices = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][stock])[['Date', 'Close']]

            # Grab only the dates of January 2024
            currentStockClosingPrices = currentStockClosingPrices[currentStockClosingPrices['Date'] >= '2024-01-01'][currentStockClosingPrices['Date'] <= '2024-01-31']

            # Rename the closing prices columns
            currentStockClosingPrices = currentStockClosingPrices.rename(columns={'Close': f'{stock}_Close'})

            # Merge the new DataFrame on the initial one
            closingPrices = pd.merge(closingPrices, currentStockClosingPrices, on='Date', how='outer')

        # Save the Final DataFrame 
        closingPrices.to_csv(closingPricesPath, sep=',', index=False)
    else:
        # Load the Previously computed DataFrame
        closingPrices = pd.read_csv(closingPricesPath)

    # Return the Final DataFrame
    return closingPrices

def getStocksVolatility(stocks:List[str], pathsConfig:dict) -> pd.DataFrame:
    """
    # Description
        -> This function enables us to collect all the volatility values from all the stocks.
    -----------------------------------------------------------------------------------------
    := param: stocks - List with the Stocks to consider.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: Pandas DataFrame with all the stock's opening prices.
    """

    # Define the path to store the real opening prices of all the stocks
    stocksVolatilityPath = pathsConfig['ExperimentalResults']['Stocks-Volatility']

    # If the DataFrame has yet to be computed
    if not os.path.exists(stocksVolatilityPath):
        # Get the first stock
        firstStock = stocks[0]

        # Initialize the DataFrame for all the data
        volatility = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][firstStock])[['Date', 'Volatility']]

        # Filter only the dates for January 2024
        volatility = volatility[volatility['Date'] >= '2024-01-01'][volatility['Date'] <= '2024-01-31']

        # Rename the closing prices columns
        volatility = volatility.rename(columns={'Volatility': f'{firstStock}_Volatility'})

        # Iterate through the available stocks
        for stock in stocks[1:]:
            # Load the next stocks volatility values
            currentStockVolatility = pd.read_csv(pathsConfig['Datasets']['Raw-Stocks-Market-Information'][stock])[['Date', 'Volatility']]

            # Filter only the dates for January 2024
            currentStockVolatility = currentStockVolatility[currentStockVolatility['Date'] >= '2024-01-01'][currentStockVolatility['Date'] <= '2024-01-31']

            # Rename the closing prices columns
            currentStockVolatility = currentStockVolatility.rename(columns={'Volatility': f'{stock}_Volatility'})

            # Merge the new DataFrame on the initial one
            volatility = pd.merge(volatility, currentStockVolatility, on='Date', how='outer')

        # Save the Final DataFrame 
        volatility.to_csv(stocksVolatilityPath, sep=',', index=False)
    else:
        # Load the Previously computed DataFrame
        volatility = pd.read_csv(stocksVolatilityPath)

    # Return the Final DataFrame
    return volatility