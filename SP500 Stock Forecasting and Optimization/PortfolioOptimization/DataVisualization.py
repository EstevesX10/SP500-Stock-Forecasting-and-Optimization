import numpy as np
import matplotlib.pyplot as plt

def plotPortfolioPerformance():
    """
    # Description
        -> This function aims to plot the Portfolio Performance over
        January 2024 based on a couple of financial metrics.
    ----------------------------------------------------------------
    """

    # --------------------


    dias = np.arange(1, 22)
    plt.figure(figsize=(10, 6))
    # plt.plot(dias, returns, marker='o', linestyle='-', color='b', label='Return on Investment')
    plt.title('Return on Investment', fontsize=16)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Return(%)', fontsize=12)
    plt.xticks(ticks=dias, rotation=45)
    plt.grid(alpha=0.3) 
    plt.legend()  

    plt.tight_layout() 
    plt.show()

    # -----------


    plt.figure(figsize=(10, 6))
    # plt.plot(dias, risk_returns, marker='o', linestyle='-', color='b', label='Risk-Adjusted Return')
    plt.title('Risk-Adjusted Return', fontsize=16)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Risk-Return(%)', fontsize=12)
    plt.xticks(ticks=dias, rotation=45)
    plt.grid(alpha=0.3) 
    plt.legend()  

    plt.tight_layout() 
    plt.show()