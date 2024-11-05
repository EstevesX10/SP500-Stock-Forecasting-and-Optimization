def loadConfig() -> dict:
    # This Funtion serves to define a configuration to be used within the Project's development
    return {'stock':'NVDA', # Selecting a Stock to be studied
            'N':3, # Number of previous data to be considered when predicting a next value
            'max_period':False, # Flag to determine whether or not to consider all the historical data, i.e, all the information available 
            'start_date':'2022-12-04', # Start date of the data to be considered
            'end_date':'2024-08-12', #  End date of the data to be considered
            'save_plot':False} # Flag that decides whether or not to save the Final graph of the Notebook