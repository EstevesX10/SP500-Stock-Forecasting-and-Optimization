from typing import (List)

def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """
    
    return {'N':3,                      # Number of previous data to be considered when predicting a next value
            'max_period':False,         # Flag to determine whether or not to consider all the historical data, i.e, all the information available 
            'start_date':'2009-10-01',  # Start date of the data to be considered
            'end_date':'2024-02-5',     # End date of the data to be considered
            'window':20,                # Window size that determines the day-interval in which to evaluate the trend of a stock (Essencially the size of the segmentation)
            'volatility_window':60,     # Window of days in which to consider to assess the volatility of a given stock within a market index fund.
            'save_plot':False}          # Flag that decides whether or not to save the Final graph of the Notebook


def createStocksMarketInformationPaths(stocks:List[str], windowed:bool) -> dict:
    """
    # Description
        -> This function helps create paths to store all the data extracted with the YFinance API.
    ----------------------------------------------------------------------------------------------
    := param: stocks - List of strings representing the Symbols for each stock.
    := param: windowed - Boolean that determines if the path corresponds to a windowed version of the extracted data or a raw version of it.
    := return: Dictionary with the paths to store the market information for the given stocks.
    """

    # Create a initial dictionary to store all the information
    stocksPaths = {}

    # Windowed Version
    if windowed:
        # Iterate through the stocks symbols
        for stock in stocks:
            stocksPaths.update({stock:f"./Datasets/Stocks/Windowed/{stock}.csv"})
    # Raw Version
    else:
        # Iterate through the stocks symbols
        for stock in stocks:
            stocksPaths.update({stock:f"./Datasets/Stocks/Raw/{stock}.csv"})

    # Return the final dictionary
    return stocksPaths

def createTrainedModelsPaths(stocks:List[str], predictionDates:List[str]) -> dict:
    """
    # Description
        -> This function creates a dictionary used to store experimental results, such as trained models.
    -----------------------------------------------------------------------------------------------------
    := param: stocks - List of strings representing the Symbols for each stock.
    := param: predictionDates - List with the dates in which we are to perform inference - January 2024.
    := return: Dictionary with the paths to store the experimental results.
    """

    # Create a initial dictionary
    modelsPaths = {}

    # Iterate through the stocks
    for stock in stocks:
        # Create a dictionary for the Stock's model paths
        stockModelPaths = {}

        # Iterate through the Prediction Dates
        for predictionDate in predictionDates:
            # Create a Dictionary to store each model path for a the current date
            modelPaths = {}
            for model in ["LGBM", "LSTM", "Prophet"]:
                # Update the modelPaths
                modelPaths.update({
                    model: f"./ExperimentalResults/TrainedModels/{stock}/{predictionDate}/{model}.pkl"
                })
            # Update the Stock Model Paths
            stockModelPaths.update({predictionDate:modelPaths})
        # Update the initial dictionary
        modelsPaths.update({stock:stockModelPaths})

    # Return the Models Paths Dictionary
    return modelsPaths

def loadInitialPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ---------------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        'ExploratoryDataAnalysis':'./ExperimentalResults/',
        'Datasets': {
            'SP500-Stocks-Wikipedia':'./Datasets/SP500-Stocks-Wikipedia.csv',
            'SP500-Market-Information':'./Datasets/SP500-Market-Information.csv',
        },
    }

def loadFinalPathsConfig(stocks:List[str], predictionDates:List[str]) -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ---------------------------------------------------------------------------------------------------------
    := param: stocks - List of strings representing the Symbols for each stock.
    := param: predictionDates - List with the dates in which we are to perform inference - January 2024.
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        'ExploratoryDataAnalysis':'./ExperimentalResults/',
        'Datasets': {
            'SP500-Stocks-Wikipedia':'./Datasets/SP500-Stocks-Wikipedia.csv',
            'SP500-Market-Information':'./Datasets/SP500-Market-Information.csv',
            'Raw-Stocks-Market-Information': createStocksMarketInformationPaths(stocks=stocks, windowed=False),
            'Windowed-Stocks-Market-Information':createStocksMarketInformationPaths(stocks=stocks, windowed=True),
        },
        'ExperimentalResults':createTrainedModelsPaths(stocks=stocks, predictionDates=predictionDates)
    }

def loadInitialSetup() -> dict:
    """
    # Description
        -> This function loads all the initial parameters 7
        to consider when performing portfolio optimization.
    -------------------------------------------------------
    := return: Dictionary with the initial configuration for the optimization phase of the project.
    """
    return {
        'initialMoney':1000,
        'buyFee':1,
        'sellFee':1,
        'limitStocksPerDay':100,
        'minimizeRisk':True,
        'maximizeRisk':False,
        'minimizeReturn':False,
        'maximizeReturn':True
    }