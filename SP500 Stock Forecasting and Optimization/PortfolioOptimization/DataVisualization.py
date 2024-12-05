from typing import (Tuple, List)
import numpy as np
import matplotlib.pyplot as plt
from .pickleFileManagement import (loadObject)

def plotPortfolioPerformance(days:List[str], portfolioEvaluations:List[float], portfolioReturns:List[float], portfolioRiskReturns:List[float]) -> None:
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
    axs[0].plot(days, portfolioEvaluations, label="$", color="#138d75")
    axs[0].set_title("Portfolio Evaluation")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Amount ($)")
    axs[0].tick_params(axis='x', rotation=45) 
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    plt.sca(axs[0])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 

    # Plot the Returns
    axs[1].plot(days, portfolioReturns, label="ROI", color="#2980b9")
    axs[1].set_title("Return on Investment")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Return (%)")
    axs[1].tick_params(axis='x', rotation=45, labelsize=7) 
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    plt.sca(axs[1])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 

    # Plot the Risk Returns
    axs[2].plot(days, portfolioRiskReturns, label="Risk Return", color="#c0392b")
    axs[2].set_title("Risk-Adjusted Return")
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel("Risk-Return (%)")
    axs[2].tick_params(axis='x', rotation=45) 
    axs[2].legend()
    axs[2].grid(alpha=0.3)
    plt.sca(axs[2])
    plt.xticks(rotation=-45, ha='left', fontsize=8) 
    
    # Add a major title
    fig.suptitle("[Portfolio Optimization] Performance Evaluation", fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plotPopulationsFitnessScore(dates:List[str], pathsConfig:dict) -> None:
    """
    # Description
        -> This function aims to plot the fitness score obtained 
        of all populations over the 50 generations.
    -----------------------------------------------------------------------
    := param: dates - List with the dates considered.
    := param:
    := return: None, since we are onlçy plotting information.
    """

    # Load all the GA instances
    geneticAlgorithms = [
        loadObject(filePath=pathsConfig['ExperimentalResults']['Genetic-Algorithm-Results'][dates[day]]['Genetic-Algorithm'])
        for day in range(len(dates))
    ]

    # Create the figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot fitness for each GA instance
    for i, ga_instance in enumerate(geneticAlgorithms):
        # Compute the normalized Fitnesses for better visualization
        fitnessHistory = np.array(ga_instance.best_solutions_fitness)
        normalizedFitness = (fitnessHistory - fitnessHistory.min()) / (fitnessHistory.max() - fitnessHistory.min())
        axs[0].plot(normalizedFitness, label=f"GA on '{dates[i]}'")

        # Compute relative improvement
        fitnessHistory = np.array(ga_instance.best_solutions_fitness)
        relativeImprovement = (fitnessHistory - fitnessHistory[0]) / fitnessHistory[0]
        axs[1].plot(relativeImprovement, label=f"GA on '{dates[i]}'")

    # Customize the first subplot
    axs[0].set_title("Normalized Fitness Across Generations")
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Normalized Fitness")
    axs[0].legend(
        loc='lower right',       # Position the legend
        ncol=2,                  # Use multiple columns
        fontsize=8,              # Set font size
        handletextpad=0.5,       # Reduce space between handle and text
        labelspacing=0.2,        # Reduce vertical spacing
        frameon=True,            # Enable frame
        framealpha=0.5,          # Add transparency to the frame
        title="GA Instances",    # Add a title
        title_fontsize=10        # Customize title font size
    )
    axs[0].grid(True)

    # Customize the second subplot
    axs[1].set_title("Relative Improvement Across Generations")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Relative Improvement")
    axs[1].legend(
        loc='lower right',       # Position the legend
        ncol=2,                  # Use multiple columns
        fontsize=8,              # Set font size
        handletextpad=0.5,       # Reduce space between handle and text
        labelspacing=0.2,        # Reduce vertical spacing
        frameon=True,            # Enable frame
        framealpha=0.5,          # Add transparency to the frame
        title="GA Instances",    # Add a title
        title_fontsize=10        # Customize title font size
    )
    axs[1].grid(True)

    # Add a major title
    fig.suptitle("[Fitness Across Generations]", fontsize=14, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
