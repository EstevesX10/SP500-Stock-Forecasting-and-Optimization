from typing import (List)
import numpy as np

def getCurrentPortfolioEvaluation(numberStocks:int, initialValue:float, investment:float, stocks:List[float], closingPrices:List[float]) -> float:
    """
    # Description
        -> This function computes the current Portfolio Evaluation based 
        on the purchased stocks and their current price evaluation.
    --------------------------------------------------------------------
    := param: numberStocks - Amount of Stocks that can be invested upon.
    := param: initialValue - Initial amount of money available to place orders for the stocks.
    := param: investment - Amount of money already utilized to place previous orders.
    := param: stocks - List with the amount of shares the client has bought.
    := param: closingPrices - List with all the Closing Prices for each Stock.
    := return: The Current Evaluation of the Investment Portfolio.
    """

    # Initialize the Initial portfolio eval to the money left over
    currentPortfolioEval = initialValue - investment

    # Iterate through the available stocks
    for i in range(numberStocks):
        # Update the portfolio evaluation based on the stocks already purchased
        currentPortfolioEval += stocks[i] * closingPrices[i]
    
    # Return the current evaluation of the Portfolio
    return currentPortfolioEval

def getMoneyInvested(numberStocks:int, stocks:List[float], openingPrices:List[float]) -> float:
    """
    # Description
        -> This function allows to compute the amount of money invested given a certain operation.
    ----------------------------------------------------------------------------------------------
    := param: numberStocks - Amount of Stocks that can be invested upon.
    := param: stocks - List with the amount of shares the client has bought.
    := param: openingPrices - List with all the Opening Prices for each Stock.
    := return: The amount of money invested of the placed order.
    """
    # Initialize the investment to 0
    investedMoney = 0

    # Iterate through all the selected stocks
    for i in range(numberStocks):
        # Update the Investment 
        investedMoney += stocks[i] * openingPrices[i]

    # Compute the amount of operations performed
    op = np.count_nonzero(stocks)

    # Apply the operation cost / fee
    investedMoney += op*2
    print(f"Operation cost: {op*2}")

    # Return the money invested on the given Operation
    return investedMoney

def getTotalReturn(initialEval:float, finalEval:float) -> float:
    """
    # Description
        -> This function computed the Total Return in Percentage (%)
        of the overall investment, at the end of January 2024.
    ----------------------------------------------------------------
    := param: initialEval - Initial Evaluation of the Investment / Portfolio. (Corresponds to the initialy available money)
    := param: finalEval - Final Evaluation of the Investment / Portfolio.
    := return: Total Return of the Investment in %.
    """

    # Compute and return the investment total return
    return ((finalEval - initialEval) / initialEval) * 100

def getROI(investment:float, earnings:float) -> float:
    """
    # Description
        -> This function computes the return of Investment of the Porfolio.
    -----------------------------------------------------------------------
    := param: investment - Amount of Money Invested in the Stock Market.
    := param: earnings - Final Evaluation of the Portfolio.
    := return: The ROI Value (In Percentage %).
    """

    # Compute and return the ROI Value
    return ((earnings - investment) / investment) * 100

def getRiskAdjustedReturn(avgReturn :float, riskFreeRate:float, avgVolatility:float) -> float:
    """
    # Description
        -> This function computes the Risk Adjusted Return.
    -------------------------------------------------------
    := param: avgReturn - Average Return Value.
    := param: riskFreeRate - Rate in which we considered the risk as low.
    := param: avgVolatility - Average Volatility of the Stock.
    := return: The Risk Adjusted Return considering the investment return, volatility and freeRiskRate.
    """

    # Compute and Return the Risk Adjusted Return
    return (avgReturn - riskFreeRate) / avgVolatility