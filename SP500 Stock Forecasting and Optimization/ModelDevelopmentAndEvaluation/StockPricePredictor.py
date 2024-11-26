from typing import (List, Tuple)
import numpy as np
import pandas as pd
import os
from pathlib import (Path)
from datetime import datetime as dt

from tensorflow.keras.models import (Sequential) # type: ignore
from tensorflow.keras.layers import (Dense, LSTM) # type: ignore

from lightgbm import (LGBMRegressor)

from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.metrics import mean_squared_error

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

    # I WANT THAT FOR EACH INSTANCE OF THIS CLASS I CAN PREDICT A GIVEN STOCK PRICE OVER THE SELECTED PREDICTION DATES INSIDE THE PASSED LIST - FOR EACH ML MODEL
    # I WANT TO COMPUTE THE MODELS AND MAYBE ADD A METHOD TO

    def checkFolder(self, path:str) -> None:
        """
        # Description
            -> This method helps ensure that all the nested
            folders inside a given model path exist.
        ---------------------------------------------------
        := param: path - Path in which we we want all the nested folder to exist on.
        := return: None, since we are only making sure that a path exists.
        """

        # Define the Folder in which to save the object
        folderPath = Path("/".join(path.split("/")[:-1]))
        
        # Check if the directory exists. If not create it
        folderPath.mkdir(parents=True, exist_ok=True)

    def trainModels(self):
        """
        # Description
            -> This method creates, trains and performs inference over all the 
            dates given in the constructor which were to predict the closing price of.
        ------------------------------------------------------------------------------
        """

        # Iterate through the dates to predict
        for dateToPredict in self.datesToPredict:
            # Define the path in which to save the current prediction date model
            lstmModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["LSTM"]
            lgbmModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["LGBM"]

            # Make sure that each model folder is created
            self.checkFolder(path=lstmModelPath)
            self.checkFolder(path=lgbmModelPath)

            # Create a Manager for the current date to predict
            stockDataManager = stockPriceManager(stockSymbol=self.stockSymbol, feature='Close', windowSize=self.config['window'], predictionDate=dateToPredict, pathsConfig=self.pathsConfig)

            # Split the Data
            X_train, y_train, X_test, y_test = stockDataManager.trainTestSplit()

            print(stockDataManager.predictionDate, type(X_train))

            # Get the Scaler Path
            scalerPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["Scaler"]

            # Load the Scaler
            scaler = loadObject(filePath=scalerPath)

            # Compute the prediction of the closing Price with the LSTM Network Architecture
            y_pred_LSTM = self.createTrainPredictLSTM(X_train=X_train, y_train=y_train, X_test=X_test, filePath=lstmModelPath)

            # Compute the prediction of the closing Price with a Light Gradient Boosting Machine
            y_pred_LGBM = self.createLGBM(X_train=X_train, y_train=y_train, X_test=X_test, filePath=lgbmModelPath)

            # Inverse Scale the predicted values
            y_test = scaler.inverse_transform([y_test])
            y_pred_LSTM = scaler.inverse_transform(y_pred_LSTM)
            y_pred_LGBM = scaler.inverse_transform([y_pred_LGBM])

            print(f"{y_test =}")
            print(f"{y_pred_LSTM =}")
            print(f"{y_pred_LGBM =}")

            # Process the Predictions - COMPUTE ERROR
            #TODO

            # break

    def createTrainPredictLSTM(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, filePath:str) -> float:
        """
        # Description
            -> This method creates, compiles, trains and predicts the closing Value 
            with help of a LSTM Network Architecture (Using a Sequential Approach).
        ---------------------------------------------------------------------------
        := param: X_train - Features of the train set.
        := param: y_train - Target Values on the train set.
        := param: X_test - Features of the test set.
        := param: filePath - Path to the computed model.
        := return: The prediction of the trained model.
        """
        
        # Check if the model has yet to be computed
        if not os.path.exists(filePath):
            # Define the Network Architecture
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(64, return_sequences=False),
                Dense(25),
                Dense(1)
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mean_absolute_error'], run_eagerly=True)

            # Train the model
            model.fit(X_train, y_train, batch_size=32, epochs=100)
        
            # Save the Model
            saveObject(objectObtained=model, filePath=filePath)
        
        # The Model has already been computed
        else:
            # Load the Model
            model = loadObject(filePath=filePath)

        # Perform Prediction
        y_pred = model.predict(X_test)

        # Return the scaled prediction
        return y_pred

    def createLGBM(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, filePath:str) -> float:
        """
        # Description
            -> This Method helps create, train and predict the closing price
            using a instance of the Light Gradient Boosting Machine Model.
        --------------------------------------------------------------------
        := param: X_train - Features of the train set.
        := param: y_train - Target Values on the train set.
        := param: X_test - Features of the test set.
        := param: filePath - Path to the computed model.
        := return: Prediction of the Closing Price which the class is working with.
        """

        # The model has not yet been computed
        if not os.path.exists(filePath):
            # Create a instance of the model
            model = LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1
            )

            # Train the Model
            model.fit(
                X_train, y_train,
                eval_metric='rmse',
            )

            # Save the Model
            saveObject(objectObtained=model, filePath=filePath)
        
        # The model has already been computed
        else:
            # Load the model
            model = loadObject(filePath=filePath)

        # Get best iteration
        best_iteration = model.best_iteration_

        # Predict using the best iteration
        y_pred = model.predict(X_test, num_iteration=best_iteration)

        # Return the scaled prediction
        return y_pred

    def createModel(self):
        raise ValueError("TO BE IMPLEMENTED!")
