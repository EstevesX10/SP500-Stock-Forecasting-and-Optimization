from typing import (Tuple)
import numpy as np
import pandas as pd
import os
from pathlib import (Path)
from datetime import datetime as dt
from sklearn.model_selection import (TimeSeriesSplit)
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)

class stockPriceManager:
    def __init__(self, stockSymbol:str, feature:str, windowSize:int, predictionDate:str, pathsConfig:dict) -> None:
        """
        # Description
            -> Constructor of a stockPriceManager which helps process the time series data
            used to train deep learning models to later help us predict the stock market prices for January 2024. 
        ---------------------------------------------------------------------------------------------------------
        := param: stockSymbol - Symbol of the Stock which we are to consider when preparing the data.
        := param: feature - Feature to perform temporal sampling from.
        := param: window_size - Size of the window used for tranning.
        := param: predictionDate - Date in which the train set stops since we want to predict it.
        := param: pathsConfig - Dictionary with important file paths used to process files within the project mainframe.
        := return: None, since we are simply instanciating a class object.
        """

        # Save given parameters
        self.stockSymbol = stockSymbol
        self.feature = feature
        self.windowSize = windowSize
        self.pathsConfig = pathsConfig

        # Define the file path in which the stock's market information resides in
        self.stockFilePath = self.pathsConfig['Datasets']['Raw-Stocks-Market-Information'][f"{stockSymbol}"]
        
        # Load the stock market history DataFrame
        self.df = self.loadStockMarketHistory()

        # Compute the prediction date [Takes into consideration getting an invalid date - It will get the closest date in the future]
        self.predictionDate = self.df['Date'].iloc[self.df.index[self.df['Date'] >= predictionDate]].tolist()[0]

        # Compute the date to use for Validation
        self.validationDate = self.df['Date'].iloc[self.df.index[self.df['Date'] >= predictionDate] - 1].tolist()[0]

        # Make a check for the window size
        if (self.windowSize + 2 > self.df.shape[0]):
            raise ValueError("Invalid Window Size Given")

        # The window size must be equal or larger than 3 - To include train, validation and test
        if (self.windowSize < 3):
            raise ValueError("Invalid Window Size [The size must be 3 or larger!]")
        
        # Check if the selected feature is inside the columns of the given DataFrame
        if (self.feature not in self.df.columns):
            raise ValueError("Invalid Feature Selected")

        # Convert the original DataFrame into a Windowed DataFrame
        self.windowed_df = self.createdWindowedStockMarketHistory()

    def loadStockMarketHistory(self) -> pd.DataFrame:
        """
        # Description
            -> This Method helps load the extracted stock market history 
            previously computed and saved inside a csv file.
        ----------------------------------------------------------------
        := return - Pandas DataFrame extracted.
        """

        # Read the Extracted DataFrame
        return pd.read_csv(self.stockFilePath)
    
    def createdWindowedStockMarketHistory(self) -> pd.DataFrame:
        """
        # Description
            -> This function helps create a windowed DataFrame with 0 to N time stamps 
            for train and 1 test stamp as the target value.
        ------------------------------------------------------------------------------
        := return: Windowed DataFrame.
        """

        # Define a filepath for the windowed DataFrame
        windowedStockDataFile = self.pathsConfig['Datasets']['Windowed-Stocks-Market-Information'][f"{self.stockSymbol}"]

        # Define the Folder in which to save the windowed DataFrame
        windowedStockDataFolder = Path("/".join(windowedStockDataFile.split("/")[:-1]))
        
        # Check if the directory exists. If not create it
        windowedStockDataFolder.mkdir(parents=True, exist_ok=True)

        if (not os.path.exists(windowedStockDataFile)):
            # Create a Variable to store all the data regarding the time segments
            data = []

            # Select the data used for train, validation and test
            trainCondition = self.df['Date'] < self.validationDate
            validationCondition = self.df['Date'] == self.validationDate
            testCondition = self.df['Date'] == self.predictionDate

            # Create a scaler to normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Normalize the Closing Price for the training set
            trainClosingPrices = self.df.loc[trainCondition, 'Close'].values.reshape(-1, 1)
            self.df.loc[trainCondition, 'Close'] = scaler.fit_transform(trainClosingPrices)

            # Normalize the Closing Price for the validation set
            validationClosingPrices = self.df.loc[validationCondition, 'Close'].values.reshape(-1, 1)
            self.df.loc[validationCondition, 'Close'] = scaler.transform(validationClosingPrices)

            # Normalize the Closing Price for the test set
            testClosingPrices = self.df.loc[testCondition, 'Close'].values.reshape(-1, 1)
            self.df.loc[testCondition, 'Close'] = scaler.transform(testClosingPrices)

            # Iterate through the DataFrame
            for index, row in self.df.iloc[:self.df.shape[0] - self.windowSize - 1, :].iterrows():
                currentTimeSequence = {}
                timeStamp = 0
                
                # Iterate through the DataFrame within the current window 
                for _, timeRow in self.df.iloc[index : index + self.windowSize, :].iterrows():
                    # Test Day
                    if timeStamp == self.windowSize - 1:
                        currentDay = 'Target'
                    # Train Days
                    else:
                        currentDay = f'Train_{timeStamp}'
                    
                    # Update the current time sequence
                    currentTimeSequence.update({currentDay:timeRow[self.feature]})
                    timeStamp += 1
                
                # Add the target date
                currentTimeSequence.update({'Target_Date':self.df.iloc[index + self.windowSize]['Date']})

                # Append the new time sequence
                data.append(currentTimeSequence)

            # Create a DataFrame with the final DataFrame
            df = pd.DataFrame(data=data)

            # Save the DataFrame into a file
            df.to_csv(windowedStockDataFile, sep=',', index=False)

        else:
            # Read the previously computed DataFrame
            df = pd.read_csv(windowedStockDataFile)

        return df

    def strToDatetime(self, string_date:str) -> dt:
        """
        # Description
            -> Converts a string of type YYYY-MM-DD into a datetime object. 
        -------------------------------------------------------------------
        := param: string_date - String that we want to convert into a datetime type object [Eg: '2003-10-10'].
        := return: Instance of Datetime based on the given date string.
        """
        # Fetching the year, month and day from the string and convert them into int
        year, month, day = list(map(int, string_date.split('-')))

        # Return a instance of datetime with the respective extracted attributes from the given string
        return dt(year=year, month=month, day=day)

    def trainTestSplit(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        # Description
            -> This method helps split the windowed DataFrame into train and test
            sets to use in order to train the machine learning algorithms.
        -------------------------------------------------------------------------
        := retuns: Train and Test Sets to use for Training.
        """

        # # Define the conditions to belong on either one of the sets
        # trainCondition = self.df['Target_Date'] < self.validationDate
        # validationCondition = self.df['Target_Date'] == self.validationDate
        # testCondition = self.df['Target_Date'] == self.predictionDate

        # # Select the data for the train, validation and test sets
        # train_df = self.df[trainCondition]
        # validation_df = self.df[validationCondition]
        # test_df = self.df[testCondition]

        # # Split the train, validation and test sets into features and target
        # X_train = train_df[train_df.columns[:-2]]
        # y_train = train_df[train_df.columns[-2]]
        
        # X_validation = validation_df[validation_df.columns[:-2]]
        # y_validation = validation_df[validation_df.columns[-2]]
        
        # X_test = test_df[test_df.columns[:-2]].to_numpy()
        # y_test = test_df[test_df.columns[-2]].to_numpy()

        # return X_train, y_train, X_validation, y_validation, X_test, y_test

        # Define the conditions to belong on either one of the sets (Train or Test)
        trainCondition = self.windowed_df['Target_Date'] < self.validationDate
        testCondition = self.windowed_df['Target_Date'] == self.predictionDate

        # Get the trainSize and define a point in which to separate the train and validation sets
        trainSize = self.windowed_df[trainCondition].shape[0]
        trainValMargin = int(0.8 * trainSize) 

        # print(trainSize, trainValMargin)
        # print(f"Train 0 - {trainValMargin} || Validation {trainValMargin + 1} - {trainSize}")

        # Select the data for the train, validation and test sets
        train_df = self.windowed_df[trainCondition].iloc[:trainValMargin]
        validation_df = self.windowed_df[trainCondition].iloc[trainValMargin:]
        test_df = self.windowed_df[testCondition]

        # Split the train, validation and test sets into features and target
        X_train = train_df[train_df.columns[:-2]].to_numpy()
        y_train = train_df[train_df.columns[-2]].to_numpy()
        
        X_validation = validation_df[validation_df.columns[:-2]].to_numpy()
        y_validation = validation_df[validation_df.columns[-2]].to_numpy()
        
        X_test = test_df[test_df.columns[:-2]].to_numpy()
        y_test = test_df[test_df.columns[-2]].to_numpy()

        return X_train, y_train, X_validation, y_validation, X_test, y_test
    
    def prepareDataForProphet(self) -> pd.DataFrame:
        """
        # Description
            -> This method aims to prepare the data to be fed to the Prophet model.
        ---------------------------------------------------------------------------
        := return: Pandas DataFrame which can be fed directly to a instance of the Prophet Model.
        """
        
        # Select the Date and Closing price Columns
        df = self.df[['Date', 'Close']]

        # Parse the date strings into datetime objects
        df['ds'] = pd.to_datetime(df['Date'])

        # Rename Closing Price into y (What we want to predict)
        df['y'] = df['Close']

        # Remove Unnecessary Columns
        df=df.drop(columns=['Date', 'Close'])

        # Return Final DataFrame
        return df
