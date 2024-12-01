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