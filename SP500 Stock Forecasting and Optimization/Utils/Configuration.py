def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """
    
    return {'N':3,                      # Number of previous data to be considered when predicting a next value
            'max_period':False,         # Flag to determine whether or not to consider all the historical data, i.e, all the information available 
            'start_date':'2008-10-01',  # Start date of the data to be considered
            'end_date':'2023-12-31',    # End date of the data to be considered
            'window':20,                # Window size that determines the day-interval in which to evaluate the trend of a stock (Essencially the size of the segmentation)
            'volatility_window':60,     # Window of days in which to consider to assess the volatility of a given stock within a market index fund.
            'save_plot':False}          # Flag that decides whether or not to save the Final graph of the Notebook


def loadPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        'ExploratoryDataAnalysis':'./ExperimentalResults/',
        'Datasets': {
            'SP500-Stocks-Wikipedia':'./Datasets/SP500-Stocks-Wikipedia.csv',
            'SP500-Market-Information':'./Datasets/SP500-Market-Information.csv',
            'Stocks-Market-Information':'./Datasets/Stocks/'
        }
    }