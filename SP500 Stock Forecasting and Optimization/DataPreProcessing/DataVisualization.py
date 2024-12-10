import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import os

def plotStockClosingPrice(stockMarketHistory:pd.DataFrame=None, title:str=None) -> None:
    """
    # Description
        -> This function helps plot the closing price of a given stock over time.
    -----------------------------------------------------------------------------
    := param: stockMarketHistory - Pandas DataFrame with the stock's market information.
    := param: title - Title for the final graph.
    := return: None, since we are only plotting a graph. 
    """

    # Check if a market history was given
    if stockMarketHistory is None:
        raise ValueError("Missing a DataFrame with the Stock's Market History!")

    # Define a default value for the title if none was given
    title = 'Closing Prices' if title is None else title

    # Create a copy of the original DataFrame
    df = stockMarketHistory.copy()

    # Defining the index as the Date Column
    df.index = df['Date']

    # Ensure the index is in DateTime format
    df.index = pd.to_datetime(df.index)

    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    # Plot the Closing Prices
    plt.plot(df.index, df['Close'], label='Closing Price', color='#29599c')

    # Compute the rolling mean and standard deviation
    rollingMean = df['Close'].rolling(window=20).mean()
    rollingStd = df['Close'].rolling(window=20).std()

    # Add a shaded area for standard deviation
    plt.fill_between(
        df.index,
        rollingMean - rollingStd,
        rollingMean + rollingStd,
        color='#a7c1cc',
        alpha=0.5,
        label="Rolling Std Dev"
    )

    # Populate the title, X and Y axis alongside a legend
    plt.title(title)
    plt.xlabel(None)
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(alpha=0.4, linestyle='dashed')
    plt.tight_layout()
    plt.show()

def plotStockStats(stockMarketHistory:pd.DataFrame=None, title:str=None) -> None:
    """
    # Description
        -> This function helps plot important features related to a given stock.
    ----------------------------------------------------------------------------
    := param: stockMarketHistory - Pandas DataFrame with the stock's market information.
    := param: title - Title for the final graph.
    := return: None, since we are only plotting a graph. 
    """

    # Check if a market history was given
    if stockMarketHistory is None:
        raise ValueError("Missing a DataFrame with the Stock's Market History!")

    # Define a default value for the title if none was given
    title = 'Statistics' if title is None else title

    # Create a copy of the original DataFrame
    df = stockMarketHistory.copy()

    # Defining the index as the Date Column
    df.index = df['Date']

    # Ensure the index is in DateTime format
    df.index = pd.to_datetime(df.index)

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))

    # Plot the Closing Prices according to the index values [In theory, the DataFrames contain the dates as their index values]
    axes[0, 0].set_title('Closing Prices')
    axes[0, 0].plot(df.index, df['Close'], label='Closing Price', color='#29599c', alpha=0.6)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot the Sales Volume
    axes[0, 1].set_title('Sales Volume')
    axes[0, 1].plot(df.index, df['Volume'], label="Sales Volume", color='#f66b6e', alpha=0.6)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot the bollinger bands
    axes[0, 2].set_title("Bollinger Bands")
    axes[0, 2].plot(df.index, df['Close'], label='Closing Price', color='#29599c', alpha=0.6)
    axes[0, 2].plot(df.index, df['UpperBB'], label='Upper Bollinger Band', color='#f66b6e', alpha=0.6)
    axes[0, 2].plot(df.index, df['LowerBB'], label='Lower Bollinger Band', color='#4cb07a', alpha=0.6)
    axes[0, 2].fill_between(df.index, df['LowerBB'], df['UpperBB'], color='gray', alpha=0.2)
    axes[0, 2].set_xlabel("Date")
    axes[0, 2].set_ylabel("Price (USD)")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Plot the Stock Daily Returns
    axes[1, 0].set_title('Daily Returns')
    axes[1, 0].plot(df.index, df['Daily_Return'], label='Daily Return', color='#29599c', alpha=0.6)
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Price (USD)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot the Stock Window Returns
    axes[1, 1].set_title('Window Returns')
    axes[1, 1].plot(df.index, df['Window_Return'], label='Window Return', color='#4cb07a', alpha=0.6)
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Price (USD)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Plot the Stock Volatility
    axes[1, 2].set_title('Volatility')
    axes[1, 2].plot(df.index, df['Volatility'], label='Volatility', color='#4cb07a', alpha=0.6)
    axes[1, 2].set_xlabel("Date")
    axes[1, 2].set_ylabel("Volatility (%)")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Add a global title
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def pastelizeColor(c:tuple, weight:float=None) -> np.ndarray:
    """
    # Description
        -> Lightens the input color by mixing it with white, producing a pastel effect.
    -----------------------------------------------------------------------------------
    := param: c - Original color.
    := param: weight - Amount of white to mix (0 = full color, 1 = full white).
    """

    # Set a default weight
    weight = 0.5 if weight is None else weight

    # Initialize a array with the white color values to help create the pastel version of the given color
    white = np.array([1, 1, 1])

    # Returns a tuple with the values for the pastel version of the color provided
    return mcolors.to_rgba((np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight))

def plotFeatureDistribution(df:pd.DataFrame=None, classFeature:str=None, forceCategorical:bool=None, pathsConfig:dict=None, featureDecoder:dict=None) -> None:
    """
    # Description
        -> This function plots the distribution of a feature (column) in a dataset.
    -------------------------------------------------------------------------------
    := param: df - Pandas DataFrame containing the dataset metadata.
    := param: feature - Feature of the dataset to plot.
    := param: forceCategorical - Forces a categorical analysis on a numerical feature.
    := param: pathsConfig - Dictionary with important paths used to store some plots.
    := param: featureDecoder - Dictionary with the conversion between the column value and its label [From Integer to String].
    """

    # Check if a dataframe was provided
    if df is None:
        print('The dataframe was not provided.')
        return
    
    # Check if a feature was given
    if classFeature is None:
        print('Missing a feature to Analyse.')
        return

    # Check if the feature exists on the dataset
    if classFeature not in df.columns:
        print(f"The feature '{classFeature}' is not present in the dataset.")
        return

    # Set default value
    forceCategorical = False if forceCategorical is None else forceCategorical

    # Define a file path to store the final plot
    if pathsConfig is not None:
        savePlotPath = pathsConfig['ExploratoryDataAnalysis'] + '/' + f'{classFeature}Distribution.png'
    else:
        savePlotPath = None

    # Define a Figure size
    figureSize = (8,5)

    # Check if the plot has already been computed
    if savePlotPath is not None and os.path.exists(savePlotPath):
        # Load the image file with the plot
        plot = mpimg.imread(savePlotPath)

        # Get the dimensions of the plot in pixels
        height, width, _ = plot.shape

        # Set a DPI value used to previously save the plot
        dpi = 100

        # Create a figure with the exact same dimensions as the previouly computed plot
        _ = plt.figure(figsize=(width / 2 / dpi, height / 2 / dpi), dpi=dpi)

        # Display the plot
        plt.imshow(plot)
        plt.axis('off')
        plt.show()
    else:
        # Check the feature type
        if pd.api.types.is_numeric_dtype(df[classFeature]):
            # For numerical class-like features, we can treat them as categories
            if forceCategorical:
                # Create a figure
                _ = plt.figure(figsize=figureSize)

                # Get unique values and their counts
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Check if a feature Decoder was given and map the values if possible
                if featureDecoder is not None:
                    # Map the integer values to string labels
                    labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
                    
                    # Tilt x-axis labels by 0 degrees and adjust the fontsize
                    plt.xticks(rotation=0, ha='center', fontsize=8)
                
                # Use numerical values as the class labels
                else:
                    labels = valueCounts.index

                # Create a color map from green to red
                cmap = plt.get_cmap('RdYlGn_r')  # Reversed 'Red-Yellow-Green' colormap (green to red)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(f'{classFeature} Labels', labelpad=20)
                plt.ylabel('Number of Samples')
                
                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()
            
            # For numerical features, use a histogram
            else:
                # Create a figure
                plt.figure(figsize=figureSize)

                # Plot the histogram with gradient colors
                plt.hist(df[classFeature], bins=30, color='lightgreen', edgecolor='lightgrey', alpha=1.0, zorder=2)
                
                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(classFeature)
                plt.ylabel('Frequency')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=0, ha='center', fontsize=10)

                # Plot the grid behind the bars
                plt.grid(True, zorder=1)
                
                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()

        # For categorical features, use a bar plot
        elif pd.api.types.is_categorical_dtype(df[classFeature]) or df[classFeature].dtype == object:
                # Create a figure
                plt.figure(figsize=figureSize)

                # Get unique values and their counts
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Create a color map from green to red
                cmap = plt.get_cmap('viridis')  # Reversed 'Red-Yellow-Green' colormap (green to red)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(valueCounts.index.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(f'{classFeature} Labels', labelpad=15)
                plt.ylabel('Number of Samples')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=25, ha='right', fontsize=8)

                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()
        
        # Unknown Behaviour
        else:
            print(f"The feature '{classFeature}' is not supported for plotting.")
