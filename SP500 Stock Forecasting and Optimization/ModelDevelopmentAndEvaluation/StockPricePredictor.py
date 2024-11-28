from typing import (List, Tuple)
import numpy as np # type: ignore
import pandas as pd # type: ignore
import os
from pathlib import (Path)
from datetime import datetime as dt

from tensorflow.keras.models import (Sequential) # type: ignore
from tensorflow.keras.layers import (Dense, LSTM) # type: ignore

from lightgbm import (LGBMRegressor) # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from xgboost import XGBRegressor # type: ignore

from sklearn.preprocessing import (MinMaxScaler, StandardScaler) # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import tensorflow as tf # type: ignore

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

    def trainModels(self) -> pd.DataFrame:
        """
        # Description
            -> This method creates, trains and performs inference on all the selected models 
            over all the dates given in the constructor which were to predict the closing price of.
        -------------------------------------------------------------------------------------------
        := return: Pandas DataFrame with the predictions of the closing prices for the selected stock.
        """

        # Iterate through the dates to predict
        for dateToPredict in self.datesToPredict:
            # Define the path in which to save the current prediction date model
            randomForestModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["RandomForest"]
            lgbmModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["LGBM"]
            xgboostModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["XGBoost"]
            lstmModelPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["LSTM"]
            
            # Make sure that each model folder is created
            self.checkFolder(path=randomForestModelPath)
            self.checkFolder(path=lgbmModelPath)
            self.checkFolder(path=xgboostModelPath)
            self.checkFolder(path=lstmModelPath)
            
            # Create a Manager for the current date to predict
            stockDataManager = stockPriceManager(stockSymbol=self.stockSymbol, feature='Close', windowSize=self.config['window'], predictionDate=dateToPredict, pathsConfig=self.pathsConfig)

            # Split the Data
            X_train, y_train, X_test, y_test = stockDataManager.trainTestSplit()

            # Get the Scaler Path
            scalerPath = self.pathsConfig["ExperimentalResults"][self.stockSymbol][dateToPredict]["Scaler"]

            # Load the Scaler
            scaler = loadObject(filePath=scalerPath)
            
            # Compute the prediction of the closing Price with a Light Gradient Boosting Machine
            y_pred_RandomForest = self.createTrainPredictRandomForest(X_train=X_train, y_train=y_train, X_test=X_test, filePath=randomForestModelPath)
            
            # Compute the prediction of the closing Price with a Light Gradient Boosting Machine
            y_pred_LGBM = self.createTrainPredictLGBM(X_train=X_train, y_train=y_train, X_test=X_test, filePath=lgbmModelPath)
            
            # Compute the prediction of the closing Price with a Light Gradient Boosting Machine
            y_pred_XGBoost = self.createTrainPredictXGBoost(X_train=X_train, y_train=y_train, X_test=X_test, filePath=xgboostModelPath)

            # Compute the prediction of the closing Price with the LSTM Network Architecture
            y_pred_LSTM = self.createTrainPredictLSTM(X_train=X_train, y_train=y_train, X_test=X_test, filePath=lstmModelPath)

            # Inverse Scale the predicted values
            y_test = scaler.inverse_transform([y_test])
            y_pred_RandomForest = scaler.inverse_transform([y_pred_RandomForest])
            y_pred_LGBM = scaler.inverse_transform([y_pred_LGBM])
            y_pred_XGBoost = scaler.inverse_transform([y_pred_XGBoost])
            y_pred_LSTM = scaler.inverse_transform(y_pred_LSTM)
    
            print(f"{y_test = }")
            print(f"{y_pred_RandomForest = }")
            print(f"{y_pred_LGBM = }")
            print(f"{y_pred_XGBoost = }")
            print(f"{y_pred_LSTM = }")
            

            # Process the Predictions - COMPUTE ERROR
            #TODO

            # mae = mean_absolute_error(y_test, y_pred)
            # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            # r2 = r2_score(y_test, y_pred)

            # break

    def createTrainPredictRandomForest(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, filePath:str) -> float:
        """
        # Description
            -> This Method helps create, train and predict the 
            closing price using a instance of Random Forest Model.
        ----------------------------------------------------------
        := param: X_train - Features of the train set.
        := param: y_train - Target Values on the train set.
        := param: X_test - Features of the test set.
        := param: filePath - Path to the computed model.
        := return: Prediction of the Closing Price which the class is working with.
        """

        # The model has not yet been computed
        if not os.path.exists(filePath):
            # Create a instance of the model
            model = RandomForestRegressor(
                n_estimators=1000,  # Number of trees
                max_depth=5,        # Maximum depth of each tree
                random_state=42,    # Reproducibility
                n_jobs=-1           # Use all available processors
            )

            # Train the Model
            model.fit(X_train, y_train)

            # Save the Model
            saveObject(objectObtained=model, filePath=filePath)
        
        # The model has already been computed
        else:
            # Load the model
            model = loadObject(filePath=filePath)

        # Predict using the best iteration
        y_pred = model.predict(X_test)

        # Return the scaled prediction
        return y_pred

    def createTrainPredictLGBM(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, filePath:str) -> float:
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
    
    def createTrainPredictXGBoost(self, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, filePath:str) -> float:
        """
        # Description
            -> This Method helps create, train and predict the 
            closing price using a instance of XGBoost Model.
        ------------------------------------------------------
        := param: X_train - Features of the train set.
        := param: y_train - Target Values on the train set.
        := param: X_test - Features of the test set.
        := param: filePath - Path to the computed model.
        := return: Prediction of the Closing Price which the class is working with.
        """

        # The model has not yet been computed
        if not os.path.exists(filePath):
            # Create a instance of the model
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                random_state=42
            )

            # Train the Model
            model.fit(X_train, y_train)

            # Save the Model
            saveObject(objectObtained=model, filePath=filePath)
        
        # The model has already been computed
        else:
            # Load the model
            model = loadObject(filePath=filePath)

        # Predict using the best iteration
        y_pred = model.predict(X_test)

        # Return the scaled prediction
        return y_pred

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
            model.fit(X_train, y_train, batch_size=32, epochs=50)
        
            # Save the Model
            model.save(filePath)
            # saveObject(objectObtained=model, filePath=filePath)
        
        # The Model has already been computed
        else:
            # Load the Model
            # model = loadObject(filePath=filePath)
            model = tf.keras.models.load_model(filePath)

        # Perform Prediction
        y_pred = model.predict(X_test)

        # Return the scaled prediction
        return y_pred
