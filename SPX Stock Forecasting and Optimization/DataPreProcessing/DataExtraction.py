import pandas as pd
import os

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
        
        # Display the first few rows of the table
        sp500Stocks.head()

        # Save the DataFrane
        sp500Stocks.to_csv(pathsConfig['Datasets']['SP500-Stocks-Wikipedia'], sep=',', index=False)

    else:
        # Load the data
        sp500Stocks = pd.read_csv(pathsConfig['Datasets']['SP500-Stocks-Wikipedia'])

    # Return the DataFrame
    return sp500Stocks