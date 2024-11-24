from typing import (List, Tuple)
import numpy as np
import pandas as pd
import os
from pathlib import (Path)
from datetime import datetime as dt
from .StockPriceManager import (stockPriceManager)
from .pickleFileManagement import (saveObject, loadObject)

class StockPricePredictor:
    def __init__(self, stockSymbol:str=None, datesToPredict:List[str]=None, config:dict=None, pathsConfig:dict=None) -> None:
        """
        # Description
            -> Constructor of the StockPricePredictor which helps 
            define, train and evaluate a model on the data provided by the dataManager.
        -------------------------------------------------------------------------------
        := param: stockSymbol - Ticker of the Stock to be predicted over the given dates to predict.
        := param: datesToPredict - List with the dates in which we want to predict the closing price of a stock.
        := param: config - Dictionary with important values to consider when processing time-series data.
        := param: pathsConfig - Dictionary with important file paths used to process files within the project mainframe.
        := return: None, since we are only initializing a class.
        """
        
        # Check if a dataManager was passed on
        if stockSymbol is None:
            raise ValueError("Missing a Stock to Predict!")

        # Check if the dates to predict were given
        if datesToPredict is None:
            raise ValueError("Missing the dates in which to Predict the Stock Closing Price!")

        # Check if the list with the dates is not empty
        if len(datesToPredict) == 0:
            raise ValueError("There are no dates to predict - Empty List!")

        # Verify if the config was given
        if config is None:
            raise ValueError("Missing the Configuration Dictionary!")
        
        # Check if the paths config was passed on
        if pathsConfig is None:
            raise ValueError("Missing a Configuration Dictionary for the Paths used inside this Project!")

        # Save the given arguments
        self.stockSymbol = stockSymbol
        self.datesToPredict = datesToPredict
        self.config = config
        self.pathsConfig = pathsConfig


    def createLSTM(self):
        raise ValueError("TO BE IMPLEMENTED!")
    
    def createModel(self):
        raise ValueError("TO BE IMPLEMENTED!")
