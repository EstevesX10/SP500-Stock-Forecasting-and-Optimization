<div align="center">

# Labs AI & DS | S&P-500

## Stock Forecasting and Optimization

</div>

<p align="center" width="100%">
    <img src="./SP500 Stock Forecasting and Optimization/Assets/StockTrading.png" width="40%" height="40%" />
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Python-038382?style=for-the-badge&logo=Python&logoColor=038382">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-038382?style=for-the-badge&logo=Jupyter&logoColor=038382">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/_INSERT_REPO_/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/_INSERT_REPO_?style=flat&logo=gitbook&logoColor=038382&label=License&color=038382">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/_INSERT_REPO_?style=flat&logo=googlecloudstorage&logoColor=038382&logoSize=auto&label=Repository%20Size&color=038382">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/stars/EstevesX10/_INSERT_REPO_?style=flat&logo=adafruit&logoColor=038382&logoSize=auto&label=Stars&color=038382">
    </a>
    <a href="https://github.com/EstevesX10/_INSERT_REPO_/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=038382&logoSize=auto&color=038382"> 
    </a>
</div>

## Project Overview

The stock market is **highly volatile and unpredictable** which makes **stock price forecasting** and **portfolio optimization** challenging tasks. Therefore, since investors seek strategies that can **provide risk-adjusted returns** efficiently.

This project aims to leverage machine learning algorithms to **predict future stock prices of the S&P-500 Market Index** and subsequently **apply optimization techniques** to identify the **optimal set of stocks** for daily investment. The stock selection process focuses on **maximizing returns and minimizing risks**, addressing real-world financial challenges.

## Project Development

### Dependencies & Execution

As a request from ou professor this project was developed using a `Notebook`. Therefore if you're looking forward to test it out yourself, keep in mind to either use a **[Anaconda Distribution](https://www.anaconda.com/)** or a 3rd party software that helps you inspect and execute it.

Therefore, for more informations regarding the **Virtual Environment** used in Anaconda, consider checking the [DEPENDENCIES.md](https://github.com/EstevesX10/_INSERT_REPO_/blob/main/DEPENDENCIES.md) file.

### Planned Work

To effectively **develop** this project, we have divided it into the following phases:

- `Data Preprocessing and Feature Engineering` - **Extract** and **process** historical stock market data. **Engineer relevant features** such as moving averages and volatility measures.

- `Data Cleaning` - Identify and **remove incongruent or invalid entries** from the stock market dataset. **Handle missing values, outliers, and inconsistencies** to ensure high-quality input data.

- `Exploratory Data Analysis (EDA)` - Conduct in-depth analysis to understand **data distributions and trends**. Derive **actionable insights** to inform **feature selection** and modeling strategies.

- `Model Development and Evaluation` - **Develop and evaluate predictive models**, including **LSTMs** (Long Short-Term Memory networks) and **LightGBM** (Light Gradient Boosting Machine). Implement a **sliding window** approach for training, where the model is **iteratively trained on data window** that moves forward by N days until reaching the end of the dataset.

- `Portfolio Optimization` - Apply **optimization techniques** such as Monte Carlo simulations, Min-Max strategies, and genetic algorithms. Optimize **portfolio selection** to balance the **trade-off between maximizing returns and minimizing risks**.

- `Results Analysis` - Analyze **optimization outcomes** in the context of financial performance metrics.

### Datasets

If you're interested in inspecting and executing this project yourself, you'll need access to all the `datasets` we've created.

<p align="center" width="100%">
    <img src="./SP500 Stock Forecasting and Optimization/Assets/Warning.png" width="15%" height="15%" />
</p>

Since GitHub has **file size limits**, we've made them all available in a Cloud Storage provided by Google Drive which you can access [here](https://drive.google.com/drive/folders/1j0vk_fECU9AtXVbL8Dczo14iwfzoCYNJ?usp=drive_link).

## Project Results

### S&P-500 Market Index

We began by examining the **key characteristics** of the `S&P-500 Market Index`, focusing specifically on:

- The **distribution of stocks** across different industries.
- The trends in **closing prices** over time.

<table width="100%">
    <thead>
        <th>
            <div align="center">
                Stock's Industry Distribution
            </div>
        </th>
        <th>
            <div align="center">
                Closing Prices
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="50%">
                <p align="center" width="100%">
                    <img src="./SP500 Stock Forecasting and Optimization/ExperimentalResults/GICS SectorDistribution.png" width="100%" height="100%" />
                </p>
            </td>
            <td width="55%">
                <p align="center" width="100%">
                    <img src="./SP500 Stock Forecasting and Optimization/ExperimentalResults/SP500-Closing-Prices.png" width="100%" height="100%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

To illustrate the **methodology** applied to the chosen stocks, we highlight `NVDA` as an example. By examining NVDA’s data, we can more clearly **demonstrate the steps involved** in analyzing and processing the information.

### NVDA Stock

#### [Exploratory Data Analysis]

Conducted additional **exploratory data analysis** on the stock's market trends through an in-depth examination of key **financial metrics**.

<p align="center" width="100%">
    <img src="./SP500 Stock Forecasting and Optimization/ExperimentalResults/NVDA-EDApng.png" width="100%" height="100%" />
</p>

#### [Models Performance]

Using a **20-day rolling window** methodology, we prepared the data to train **several machine learning models**, achieving the following performance **results**:

<p align="center" width="100%">
    <img src="./SP500 Stock Forecasting and Optimization/ExperimentalResults/NVDA-Models-Performace.png" width="100%" height="100%" />
</p>

### Final Portfolio Performance

Finally, leveraging a `genetic algorithm`, we carried out **portfolio optimization** to devise an asset allocation plan. This approach resulted in a **profit** of approximately **$30**, as demonstrated through various financial metrics.

<p align="center" width="100%">
    <img src="./SP500 Stock Forecasting and Optimization/ExperimentalResults/Final-Portfolio-Evalutaion.png" width="100%" height="100%" />
</p>

Overall, we have developed a **tool** designed to **assist investors in effectively managing their assets**, aiming to support them in making **informed investment decisions**.

## Authorship

- **Authors** &#8594; [Francisco Macieira](https://github.com/franciscovmacieira), [Gonçalo Esteves](https://github.com/EstevesX10) and [Nuno Gomes](https://github.com/NightF0x26)
- **Course** &#8594; Laboratory of AI and DS [[CC3044](https://sigarra.up.pt/fcup/en/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=546533)]
- **University** &#8594; Faculty of Sciences, University of Porto

<div align="right">
<sub>

<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>

</div>
