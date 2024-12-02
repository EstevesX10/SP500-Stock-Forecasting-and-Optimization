import pandas as pd
import matplotlib.pyplot as plt

def plotStocksRawResults(stockSymbol:str, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps plot each model's prediction performance
        through the market open days of January 2024.
    -------------------------------------------------------------------
    := param: stockSymbol - Symbol which we want to plot the predictors performance of.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: None, since we are only plotting the data.
    """
    
    # Define the path of the stocks raw results
    stockResultsPath = pathsConfig["ExperimentalResults"][stockSymbol][f"{stockSymbol}-Raw-Predictions"]

    # Load the DataFrame with the results
    stockResults = pd.read_csv(stockResultsPath)

    # Get the prediction columns 
    predictionColumns = [col for col in stockResults.columns if 'Prediction' in col]

    # Get the MAE columns
    maeColumns = [col for col in stockResults.columns if 'Error' in col and not '(%)' in col]

    # Get the MAPE columns
    mapeColumns = [col for col in stockResults.columns if 'Error' in col and '(%)' in col]

    # Define a Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

    # Plot the Model Predictions
    for modelPredictionColumn in predictionColumns:
        # Get the Model Name
        modelName = modelPredictionColumn.split('[')[1].split(']')[0]

        # Plot the Model Prediction
        axes[0].plot(stockResults['Date'], stockResults[modelPredictionColumn], label=modelName, marker='o')

    # Plot the Target Value
    axes[0].plot(stockResults['Date'], stockResults['Target'], label='Real Value', marker='x')

    # Set the title and the Axis labeling
    axes[0].set_title(f'{stockSymbol} Stock Value Predictions')
    axes[0].set_xlabel('Date')
    axes[0].set_xticklabels(stockResults['Date'], fontsize=7, rotation=45, ha='center', va='top')
    axes[0].set_ylabel('Stock Value')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the Mean Absolute Errors
    for maeColumn in maeColumns:
        # Get the Model Name
        modelName = maeColumn.split('[')[1].split(']')[0]

        # Plot the Model Mean Absolute Error
        axes[1].plot(stockResults['Date'], stockResults[maeColumn], label=modelName, marker='s')
    
    # Plot the title and the axis names
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Date')
    axes[1].set_xticklabels(stockResults['Date'], fontsize=7, rotation=45, ha='center', va='top')
    axes[1].set_ylabel('Error')
    axes[1].legend()
    axes[1].grid(True)

    # Plot the Mean Absolute Percentage Error
    for mapeColumn in mapeColumns:
        # Get the Model Name
        modelName = mapeColumn.split('[')[1].split(']')[0]

        # Plot the Model Mean Absolute Percentage Error
        axes[2].plot(stockResults['Date'], stockResults[mapeColumn], label=modelName, marker='s')
    
    # Set the title and the Axis labeling
    axes[2].set_title('Mean Absolute Percentage Error')
    axes[2].set_xlabel('Date')
    axes[2].set_xticklabels(stockResults['Date'], fontsize=7, rotation=45, ha='center', va='top')
    axes[2].set_ylabel('Error')
    axes[2].legend()
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    plt.show()