<div align="center">
  <img src="images/readme_cover.png" width="100%"
       style="margin-left:auto;margin-right:auto;display:block;">
</div>

# **Forecasting Upward and Downward Movements in Petrobras Stock Using Machine Learning**

> In this repository, we apply **Machine Learning** techniques to predict the directional behavior of **Petrobras (PETR4)** daily returns. A rolling-window framework is used to train, evaluate, and deploy multiple classification models, including **Logistic Regression, SVC, Gradient Boosting, and K-Nearest Neighbors**, generating sequential out-of-sample predictions from **2010 to 2023**. The resulting predictions are evaluated through classification metrics and a financial backtest to investigate their predictive and economic performance.


## **1. Research Question**

Can classical machine learning models extract predictive information from historical PETR4 market data to classify the direction of the next daily return?
Rather than evaluating models using a single static train/test split, this project uses a **rolling-window framework** that repeatedly trains, evaluates, selects, and deploys models as new market data become available.
The research investigates three main questions:

1. Can directional information be extracted from historical PETR4 data?
2. How stable is model performance across different market periods?
3. Does the resulting directional signal translate into economically meaningful out-of-sample performance?

---

## 2. Experimental Setup

The experiment investigates the prediction of the directional behavior of Petrobras (PETR4) daily stock returns using Machine Learning techniques. The framework is designed to provide a reproducible and realistic evaluation through sequential out-of-sample predictions.

### 2.1 Overview of the Experimental Framework

The experimental framework consists of the following stages:

- Historical data collection
- Data preprocessing and return computation
- Return categorization
- Rolling window segmentation
- Machine Learning pipeline
- Model evaluation and selection
- Out-of-sample production

### 2.2 Data Collection

The dataset consists of daily OHLC price data for Petrobras (PETR4), traded on B3, covering the period from 2010 to 2023.
Data were obtained through the TradingView API and validated before analysis to ensure consistency and reliability.

![Historical price series of PETR4](images/asset_market_data.png)

### 2.3 Return Computation and Labeling

Daily returns were computed from consecutive closing prices:

$$
r_t = \frac{P_t-P_{t-1}}{P_{t-1}}\times100
$$

Returns were then categorized into two classes:

- **+1:** positive return
- **−1:** non-positive return

This transforms the problem into a binary classification task.

### 2.4 Rolling Window Framework

A rolling window (roll-forward) framework was employed to simulate sequential model deployment under evolving market conditions.

Each window consists of two phases:

- **Development Phase:** training and testing of candidate models.
- **Production Phase:** out-of-sample prediction using the selected model.

The development period spans approximately ten years, followed by a twelve-month production period. After each production period, the window advances and the process is repeated, generating sequential out-of-sample evaluations from 2010 to 2023.

![Rolling window framework](images/rolling_window.png)

### 2.5 Machine Learning Pipeline

The Machine Learning pipeline was implemented in Python using pandas, numpy, and scikit-learn, following principles inspired by the CRISP-DM framework.

The main stages are:

- **Load Database:** Load OHLC data and categorized returns.
- **Compute Output Feature:** Define the one-period-ahead categorized return.
- **Compute Input Features:** Generate technical and statistical features, including proprietary indicators.
- **Model Configuration:** Load modeling parameters from `modeling_arguments.json`.
- **Model Training:** Train multiple classification models.
- **Model Evaluation:** Evaluate models on training and testing datasets.
- **Model Selection:** Select the best-performing model according to the evaluation metrics.
- **Production Deployment:** Save the selected model and generate out-of-sample predictions.

### 2.6 Classification Models

The following Machine Learning classifiers were evaluated within each rolling window:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

Model performance was evaluated using:
- Accuracy
- Precision
- Recall
The selected model was subsequently stored and used during the production phase, with continuous monitoring of its performance.


## **5. Results**

### **5.1. Production Classification Performance**

Fourteen models were deployed during the 2010–2023 evaluation period.

| Metric              |    Average |
| ------------------- | ---------: |
| Production Accuracy | **53.20%** |
| Precision — Down    | **52.41%** |
| Recall — Down       | **48.67%** |
| Precision — Up      | **53.79%** |
| Recall — Up         | **57.43%** |

Thirteen of the fourteen production models achieved accuracy above 50%.

The observed production accuracy therefore remained modest, but was consistently above the 50% reference level across most rolling windows.

---

### **5.2. Model Behavior Across Rolling Windows**

The individual production results were:

| Research | Model               | Production |   Test |  Train |
| -------- | ------------------- | ---------: | -----: | -----: |
| 1        | Logistic Regression |     54.66% | 52.44% | 53.86% |
| 2        | Logistic Regression |     52.61% | 51.81% | 54.44% |
| 3        | Gradient Boosting   |     49.19% | 51.81% | 68.46% |
| 4        | KNN                 |     55.65% | 54.03% | 68.48% |
| 5        | KNN                 |     50.81% | 52.73% | 69.83% |
| 6        | Gradient Boosting   |     51.63% | 52.12% | 72.45% |
| 7        | SVC                 |     53.41% | 51.82% | 52.83% |
| 8        | SVC                 |     51.63% | 54.75% | 51.77% |
| 9        | Gradient Boosting   |     55.10% | 52.93% | 74.18% |
| 10       | KNN                 |     56.05% | 52.43% | 68.62% |
| 11       | Logistic Regression |     54.62% | 52.32% | 52.45% |
| 12       | KNN                 |     54.66% | 56.97% | 69.18% |
| 13       | Gradient Boosting   |     53.20% | 55.76% | 71.42% |
| 14       | SVC                 |     51.61% | 54.03% | 53.03% |

The results also illustrate differences between in-sample and out-of-sample behavior. In particular, Gradient Boosting reached training accuracies above 70% in several windows while producing substantially lower production accuracy in some periods.

---

## **6. Cumulative Performance**

In addition to classification metrics, the directional predictions were converted into a trading strategy to investigate their economic implications.

Correct and incorrect predictions were accumulated over time to visualize the consistency of the directional signal.

<div align="center">
  <img src="images/cumulative_predictions.png" width="90%"
       style="margin-left:auto;margin-right:auto;display:block;">
  <p align="center"><strong>Figure 1. Cumulative correct and incorrect predictions, 2010–2023.</strong></p>
</div>

A second backtest converts the directional predictions into cumulative financial returns, incorporating a **0.05% transaction cost per operation**.

<div align="center">
  <img src="images/cumulative_returns.png" width="90%"
       style="margin-left:auto;margin-right:auto;display:block;">
  <p align="center"><strong>Figure 2. Cumulative strategy returns, 2010–2023.</strong></p>
</div>

---

## **7. Annual Performance**

The resulting strategy was compared with the underlying PETR4 asset on an annual basis.

| Year | Strategy |   PETR4 |
| ---- | -------: | ------: |
| 2010 |   50.51% | −22.50% |
| 2011 |    7.94% | −14.30% |
| 2012 |   19.91% |  −2.70% |
| 2013 |   47.72% |  −6.20% |
| 2014 |   65.44% | −36.70% |
| 2015 |   56.00% | −13.90% |
| 2016 |   89.44% |  95.10% |
| 2017 |   14.29% |  18.20% |
| 2018 |   74.36% |  54.60% |
| 2019 |   32.75% |  31.70% |
| 2020 |   77.44% |  18.10% |
| 2021 |   29.07% |  30.50% |
| 2022 |   30.62% |  36.00% |
| 2023 |    2.15% |  79.80% |

The annual results show that strategy behavior varied substantially across market conditions. The strategy generated positive returns during several years in which PETR4 experienced negative annual returns, while capturing substantially less of the asset's appreciation during some strong bullish periods, particularly 2023.

---

## **8. Risk-Adjusted Performance**

| Metric                | ML Strategy |  PETR4 |
| --------------------- | ----------: | -----: |
| Average Annual Return |  **42.69%** | 19.12% |
| Volatility            |  **27.43%** | 38.69% |
| Sharpe Ratio          |    **1.08** |   0.16 |

These figures summarize the historical backtest under the assumptions described in the methodology.

They should be interpreted together with the classification results, transaction-cost assumptions, and the limitations of the experimental design.

---

## **9. Main Findings**

The experiment produced several observations:

* Directional classification accuracy averaged approximately **53.2%** across production windows.
* **13 of 14** deployed models achieved production accuracy above 50%.
* KNN showed relatively stable production behavior across several rolling windows.
* Gradient Boosting displayed substantial differences between training and production performance in some windows.
* Upward movements showed higher average recall than downward movements.
* The strategy generated materially different annual return profiles from the underlying asset.
* The backtest produced a Sharpe ratio of **1.08** under the stated assumptions.

The results indicate that the relationship between classification performance and financial performance is not one-to-one: relatively small differences in directional accuracy can produce materially different cumulative outcomes depending on the sequence and magnitude of returns.

---

## **10. Limitations**

Several limitations should be considered when interpreting the results:

* The experiment focuses on a **single equity**, PETR4.
* The feature set includes proprietary indicators that require further independent validation.
* The classification edge is relatively small in absolute terms.
* Transaction costs are represented using a fixed assumption.
* The backtest does not model market impact or liquidity constraints.
* Performance may depend on the selected rolling-window configuration.
* The study covers a historical period and does not establish future predictive performance.

---

## **11. Project Structure**

```text
📦 quantyma-petr4-ml
┣ 📂 data
┣ 📂 images
┣ 📂 notebooks
┃ ┣ 📜 01-data-exploration.ipynb
┃ ┣ 📜 02-feature-engineering.ipynb
┃ ┣ 📜 03-model-development.ipynb
┃ ┣ 📜 04-rolling-window.ipynb
┃ ┗ 📜 05-backtesting.ipynb
┣ 📂 src
┃ ├ 📜 data.py
┃ ├ 📜 features.py
┃ ├ 📜 models.py
┃ ├ 📜 rolling_window.py
┃ └ 📜 backtest.py
┣ 📜 README.md
┣ 📜 requirements.txt
┗ 📜 LICENSE
```

---

## **12. Reproducibility**

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
```

The notebooks reproduce the main stages of the research, from data processing and feature construction to model evaluation and backtesting.

---

## **13. References**

1. Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: An analysis and review*. International Journal of Forecasting, 16(4), 437–450.
2. Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. The Annals of Statistics, 29(5), 1189–1232.
3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.
4. Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
5. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression*. Wiley.
6. Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Pearson.
7. Shearer, C. (2000). *The CRISP-DM model: The new blueprint for data mining*.
8. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
