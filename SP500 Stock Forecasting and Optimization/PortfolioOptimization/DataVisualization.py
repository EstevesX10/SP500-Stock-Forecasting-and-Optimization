from typing import (Tuple, List)
import matplotlib.pyplot as plt

def plotPortfolioPerformance(days:List[str], values:List[float], returns:List[float], riskReturns:List[float]) -> None:
    """
    # Description
        -> This function aims to plot the Portfolio Performance over
        January 2024 based on a couple of financial metrics.
    ----------------------------------------------------------------
    := param: days - List with the Days in which the optimization was performed on.
    := param: values - List with the Portfolio Evaluations over January 2024.
    := param: returns - List with the Returns obtained throughout January 2024
    := param: riskReturns - List with the Risks Associated with the obtained Returns.
    := return: None, since we are simply plotting data.
    """

    # Creating the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Plot the Returns
    axs[0].plot(days, values, label="$")
    axs[0].set_title("Portfolio Evaluation")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Amount ($)")
    axs[0].tick_params(axis='x', rotation=45) 
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    plt.sca(axs[0])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 

    # Plot the Returns
    axs[1].plot(days, returns, label="Return")
    axs[1].set_title("Return on Investment")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Return (%)")
    axs[1].tick_params(axis='x', rotation=45, labelsize=7) 
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    plt.sca(axs[1])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 

    # Plot the Risk Returns
    axs[2].plot(days, riskReturns, label="Risk Return", color="orange")
    axs[2].set_title("Risk-Adjusted Return")
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel("Risk-Return (%)")
    axs[2].tick_params(axis='x', rotation=45) 
    axs[2].legend()
    axs[2].grid(alpha=0.3)
    plt.sca(axs[2])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 
    
    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()