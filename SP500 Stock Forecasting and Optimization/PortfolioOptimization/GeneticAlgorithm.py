from typing import (Tuple, List)
import numpy as np
import pandas as pd
import time
import random
from functools import (partial)
import pygad
from pygad import (GA)
from .utils import (checkFolder)
from .pickleFileManagement import (saveObject)
from .jsonFileManagement import (dictToJsonFile)
from .FinancialMetrics import (getCurrentPortfolioEvaluation, getMoneyInvested, getTotalReturn, getROI, getRiskAdjustedReturn)

def fitness_func(ga_instance:GA, solution, sol_idx:int) -> float:
        """
        # Description
            -> Fitness Function to be utilized when training the genetic algorithm.
        ---------------------------------------------------------------------------
        := param: ga_instance
        := param: solution
        := param: sol_idx
        := return: Fitness Score.
        """

        # Unpack the previously added features
        day = ga_instance.day
        numberStocks = ga_instance.numberStocks
        stocksPredictedClosingPrices = ga_instance.stocksPredictedClosingPrices
        stocksOpeningPrices = ga_instance.stocksOpeningPrices
        current_value = ga_instance.current_value
        riskFreeRate = ga_instance.riskFreeRate

        # Get the stocks expected closing prices for the selected day
        exp_prices = stocksPredictedClosingPrices.iloc[day].values 
        
        # Get the stock Opening Prices for the selected day
        prices = stocksOpeningPrices.iloc[day].values
        
        # Define the weights based on the current solution
        weights = np.array(solution, dtype=float)

        # Define the initial money invested, the gains and the amount of stocks invested in
        invested = 0
        gains = 0
        n_stocks = 0
        
        # Iterate through all the selected stocks
        for i in range(numberStocks):
            # If the expected price is lower than the opening price, then we do not consider the stock for the current day
            if (exp_prices[i] - prices[i] < 0):
                weights[i] = 0
            # If the stock's weights are not zero - The predicted value is higher than the opening price of the stock, 
            # then there is a investment opportunity
            if (weights[i] != 0):
                # Update the amount of stocks invested in
                n_stocks += 1
            
            # Update the money invested and the potential gains
            invested += weights[i] * prices[i]
            gains += weights[i] * (exp_prices[i] - prices[i])

        # Apply the Operation(s) cost
        op = np.count_nonzero(weights)
        invested += op * 2
        gains -= op * 2

        # Check if we have any money left or already invested it all or invested on the maximum amount of stocks per day
        if invested < 0 or invested > current_value or n_stocks > 20:
            return -np.inf

        # Update the best weights
        ga_instance.best_weights = weights
        
        # Return weighted gains
        return gains - riskFreeRate * np.mean(weights)  

def on_generation(ga_instance:GA) -> None:
    """
    # Description
        -> This function helps keep track of the genetic algorithm performance.
    ---------------------------------------------------------------------------
    := param: ga_instance - Instance of the Genetic Algorithm
    := return: None, since we are only showcasing information.
    """

    print(f"Generation {ga_instance.generations_completed} complete") 
    print(f"Fitness of the best solution: {ga_instance.best_solution()[1]}")
    time.sleep(1)

def dailyInvestment (numberStocks:int, day:int, predictionDates:List[str], current_value:float,
                     riskFreeRate:float, portfolioEvaluations:List[float], portfolioReturns:List[float], portfolioRiskReturns:List[float],
                     stocksOpeningPrices:pd.DataFrame, stocksClosingPrices:pd.DataFrame,
                     stocksPredictedClosingPrices:pd.DataFrame, stocksVolatility:pd.DataFrame, pathsConfig:dict) -> Tuple[List[float], List[float], List[float]]:
    """
    # Desctiption
        -> This function performs daily investmente based on the computed 
        stock's Closing Price Predictions and with help of a genetic algorithm.
    ---------------------------------------------------------------------------
    := param: numberStocks - number of selected Stocks.
    := param: day - i-th Day of January in which to analyse / perform the daily investment upon.
    := param: predictionDates - List of the Dates to consider.
    := param: riskFreeRate - Rate in which to consider a free risk scenario.
    := param: portfolioEvaluations - Portfolio Evaluations of the already passed days.
    := param: portfolioReturns - Percentage of the Portfolio Retuns over the already passed days.
    := param: portfolioRiskReturns - Percentage of the Portfolio Risk Retuns over the already passed days
    := param: stocksOpeningPrices - Opening Prices for each selected stock on January 2024.
    := param: stocksClosingPrices - Closing Prices for each selected stock on January 2024.
    := param: stocksPredictedClosingPrices - Predicted Closing Prices for each selected stock on January 2024.
    := param: stocksVolatility - Computed Values for the stock's Volatility on January 2024.
    := param: pathsConfig - Dictionary used to manage file paths.
    := return: Updated Lists for the values, returns and riskReturns.
    """
    
    # Drop the date Column from the given DataFrames
    stocksOpeningPrices = stocksOpeningPrices.drop(columns=['Date'])
    stocksClosingPrices = stocksClosingPrices.drop(columns=['Date'])
    stocksPredictedClosingPrices = stocksPredictedClosingPrices.drop(columns=['Date'])
    stocksVolatility = stocksVolatility.drop(columns=['Date'])

    # Filter the given DataFrames for the selected day
    exp_prices = stocksPredictedClosingPrices.iloc[day].values 
    prices = stocksOpeningPrices.iloc[day].values
    vol = stocksVolatility.iloc[day].values
    
    indices = []
    for i in range (numberStocks):
        try:
            threshold = np.percentile(stocksVolatility.iloc[:day].values, 25)
        except:
            threshold = 0.030
        if (exp_prices[i] - prices[i] > 0 and vol[i] < threshold):
            indices.append(i)

    indices = random.sample(indices, min(5, len(indices)))
    gene_space = [{'low': 0.00, 'high': 20.00} if i in indices else {'low': 0.00, 'high': 0.00} for i in range(numberStocks)]

    # Define a instance of the Genetic Algorithm
    ga_instance = pygad.GA(
        num_generations=50,             # Number of iterations of the genetic model
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=55,                   # Number of copanies to analyse
        gene_type=float,
        init_range_low=0.00,            # Minimum number of stocks bought of a company
        init_range_high=20.00,          # Maximum number of stocks bought of a company
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=70,     # Percentage of the weights that can be altered at each iteraction
        on_generation=on_generation,
    )

    # Attach custom data as attributes to the GA instance
    ga_instance.numberStocks = numberStocks
    ga_instance.day = day
    ga_instance.current_value = current_value
    ga_instance.riskFreeRate = riskFreeRate
    ga_instance.stocksOpeningPrices = stocksOpeningPrices
    ga_instance.stocksPredictedClosingPrices = stocksPredictedClosingPrices
    
    # Run the Genetic Algorithm
    ga_instance.run()
    
    # Ensure the path to the results exists
    checkFolder(path=pathsConfig['ExperimentalResults']['Genetic-Algorithm-Results'][predictionDates[day]]['Genetic-Algorithm'])

    # Save the Genetic Algorithm
    saveObject(objectObtained=ga_instance, filePath=pathsConfig['ExperimentalResults']['Genetic-Algorithm-Results'][predictionDates[day]]['Genetic-Algorithm'])

    # Plot the fitness variation
    ga_instance.plot_fitness()

    # Update the initial value
    initial_value = current_value

    # Update the current solution
    solution = ga_instance.best_weights

    # Get Financial Metrics
    investment = getMoneyInvested(numberStocks=numberStocks, stocks=solution, openingPrices=prices)
    current_value = getCurrentPortfolioEvaluation(numberStocks=numberStocks, initialValue=initial_value, stocks=solution, investment=investment, closingPrices=stocksClosingPrices.iloc[day].values)
    total_returns = getTotalReturn(initialEval=initial_value, finalEval=current_value)
    ret_on_inv = getROI(investment=investment, earnings=(investment + current_value - initial_value))
    risk_ret = getRiskAdjustedReturn(avgReturn=total_returns, riskFreeRate=riskFreeRate, avgVolatility=np.mean(vol))
    
    # Update the values, returns and riskReturns
    portfolioEvaluations.append(current_value)
    portfolioReturns.append(ret_on_inv)
    portfolioRiskReturns.append(risk_ret)

    # Print the Results
    print(f"Day: {day}")
    print(f"Inicial value: {initial_value}")
    print(f"Current value: {current_value}")
    print(f"Actions Investment: {solution}")
    print(f"Invested Money: {investment:.2f}")
    print(f"Total Return: {total_returns:.2f}%")
    print(f"Return on Investment: {ret_on_inv:.2f}%")
    print(f"Risk-Adjusted Return: {risk_ret:.2f}%")
    print()

    # Define a dictionary with the results
    results = {
        f"Day": day,
        f"Inicial value": initial_value,
        f"Current value": current_value,
        f"Actions Investment": solution,
        f"Invested Money": np.round(investment, decimals=2),
        f"Total Return": np.round(total_returns, decimals=2),
        f"Return on Investment": np.round(ret_on_inv, decimals=2),
        f"Risk-Adjusted Return": np.round(risk_ret, decimals=2)
    }

    # Save the Results
    dictToJsonFile(dictionary=results, filePath=pathsConfig['ExperimentalResults']['Genetic-Algorithm-Results'][predictionDates[day]]['Results'])

    # Return updated lists
    return portfolioEvaluations, portfolioReturns, portfolioRiskReturns