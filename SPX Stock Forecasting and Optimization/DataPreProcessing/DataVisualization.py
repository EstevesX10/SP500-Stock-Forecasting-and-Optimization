import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotStockClosingPrice(stockMarketHistory:pd.DataFrame=None, title:str=None) -> None:
    """
    # Description
        -> This function helps plot the closing price of a given stock over time.
    -----------------------------------------------------------------------------
    := param: stockMarketHistory - Pandas DataFrame with the stock's market information.
    := param: title - Title for the final graph.
    := return: None, since we are only plotting a graph. 
    """

    # Check if a market history was given
    if stockMarketHistory is None:
        raise ValueError("Missing a DataFrame with the Stock's Market History!")

    # Define a default value for the title if none was given
    title = 'Closing Prices' if title is None else title

    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    # Plot the Closing Prices according to the index values [In theory, the DataFrames contain the dates as their index values]
    stockMarketHistory['Close'].plot()

    # Populate the title, X and Y axis alongside a legend
    plt.title(title)
    plt.xlabel(None)
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()