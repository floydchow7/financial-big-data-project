# EPFL FIN-525: Financial Big Data Project 

**Trading on Elon Musk's mood: A sentiment-driven arbitrage strategy on Dogecoin and TSLA based on local lagged transfer entropy**


## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Result Reproduction](#result_reproduction)
6. [Authors](#authors)


## Introduction

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

3. **Install dependencies from [requirements.txt](requirements.txt)**:
```
pip install -r requirements.txt
```

**NOTE**: The needed package `pyinform` supports **Linux**, **OS X** and **Windows**. Please run this project in these systems to avoid unsupported error from the package.

4. **Ensure to have downloaded all the necessary data**: 

    4.1. Download the [FNSPID Financial News](https://huggingface.co/datasets/Zihan1004/FNSPID) dataset from Huggingface, or download from the [Switch Drive](https://www.switch.ch/en/drive).

    4.2. Download the [Crypto News+](https://www.kaggle.com/datasets/oliviervha/crypto-news) and the [Elon Musk Tweets](https://www.kaggle.com/datasets/gpreda/elon-musk-tweets) dataset from Kaggle.


    4.3. Extract the TSLA historical price data and Dogecoin historical data from the [Price_Extraction.ipynb](Price_Extraction.ipynb). Make sure that you have an available [Binance API](https://www.binance.com/en/binance-api) key before you run the notebook.

5. **Make sure the dataset is structured in the following way:**


## Result Reproduction

Run the notebook [`main_analysis.ipynb`](main_analysis.ipynb) for result reproduction. 

Noted that due to the number of the CPU cores of the machine, the runtime of the code may vary.

## Authors

The authors of the project are: 

- Mert Ülgüner ([@mulguner](https://github.com/mulguner))
- Zhuofu Zhou ([@floydchow7](https://github.com/floydchow7))
