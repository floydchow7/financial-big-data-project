# EPFL FIN-525: Financial Big Data Project 

**Trading on News and Tweets: A sentiment-driven arbitrage strategy on Dogecoin and TSLA based on local lagged transfer entropy**

This is the github repository for the project from EPFL FIN-525: Financial Big Data.

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Result Reproduction](#result_reproduction)
6. [Authors](#authors)


## Introduction

This repository contains the implementation of a sentiment-driven arbitrage trading strategy based on local lagged transfer entropy. The goal is to develop a profitable trading strategy by modeling the directed information flow from media sentiment scores to asset prices. The strategy is built by defining a set of threshold and lag parameters, which are optimized by maximizing the Sharpe ratio through a rolling optimization process.

The code provides frameworks for analyzing four distinct cases of directed information flow, each with its own independent optimization approach. The effectiveness of the strategy is evaluated by calculating cumulative returns over rolling test sets, assessing both profitability and robustness. This repository offers the full implementation to replicate and test the strategy.

## Getting Started

**Before running the project, please ensure you:**

1. **Clone this repository repository**:

```bash
git clone https://github.com/floydchow7/financial-big-data-project
cd financial-big-data-project
```

2. **Set up a virtual enviornment**:
```
python3 -m venv myenv
source myenv/bin/activate
```

If you're using conda, refer to the following code:

```
conda create -n myenv python=3.12
conda activate myenv
```

3. **Install dependencies from [requirements.txt](requirements.txt)**:
```
pip install -r requirements.txt
```

**NOTE**: The needed package `pyinform` supports **Linux**, **OS X** and **Windows**. Please run this project in these systems to avoid unsupported error from the package.


4. **Ensure to have downloaded all the necessary data**: 


    4.1. Download the [FNSPID Financial News](https://huggingface.co/datasets/Zihan1004/FNSPID) dataset from Huggingface, or download it directly from the [Switch Drive](https://drive.switch.ch/index.php/s/85mUcKuNC6OtjaE). Move the full data to `data/raw/` folder.


    4.2. Make sure that you have an available [Binance API](https://www.binance.com/en/binance-api) key before you run the notebook.

    4.3 Make sure your codebase is structured in the following way:

    ```bash
    financial-big-data-project
        ├─data
        │  ├─clean
        │  ├─processed
        │  ├─quotes
        │  └─raw
        ├─models
        │  └─classifier.py
        ├─utils
        │  ├─calibration.py
        │  ├─constants.py
        │  ├─data_extractor.py
        │  ├─helpers.py
        │  ├─pipelines.py
        │  └─visualization.py
        ├─1.data_extractor.ipynb
        ├─2.EDA.ipynb
        ├─3.main_analysis.ipynb
        ├─4.Effect_of_parallelization.ipynb
        └─5.appendix_volatility_trading.ipynb
    ```
    4.4 Change the constant parameters in [`utils/constants.py`](utils/constants.py).

    4.5 (Optional) Run the notebook [`1.data_extractor.ipynb`](1.data_extractor.ipynb) to extract the dataset. Noted that the repo has provided the needed data for analysis in the `data` folder. You **DO NOT** need to run this notebook if you would not like to run from stratch.



## Result Reproduction

Run the notebook [`3.main_analysis.ipynb`](3.main_analysis.ipynb) for result reproduction. 

**Important**: The `Switch` argument controls whether to run the full analysis or a demo workflow based on TSLA stock price. The `Switch` argument is **by default** set to False, which allows the demo workflow to display. 

Noted that due to the number of the CPU cores of the machine, the runtime of the code may vary.

## Authors

The authors of the project are: 

- Mert Ülgüner ([@mulguner](https://github.com/mulguner))
- Zhuofu Zhou ([@floydchow7](https://github.com/floydchow7))
- Hamza Karime ([@Crqbu](https://github.com/Crqbu))
