def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """
    
    return {'N':3,                      # Number of previous data to be considered when predicting a next value
            'max_period':True,         # Flag to determine whether or not to consider all the historical data, i.e, all the information available 
            'start_date':'2022-12-04',  # Start date of the data to be considered
            'end_date':'2024-08-12',    #  End date of the data to be considered
            'save_plot':False}          # Flag that decides whether or not to save the Final graph of the Notebook


def loadPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        'Datasets': {
            'SP500-Stocks-Wikipedia':'./Datasets/SP500-Stocks-Wikipedia.csv',
            'SP500-Market-Information':'./Datasets/SP500-Market-Information.csv',
            'Stocks-Market-Information':'./Datasets/Stocks/'
        }
    }