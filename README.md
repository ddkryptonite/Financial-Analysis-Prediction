# Pharmaceutical Financial Data Analysis and Forecasting with Random Forest

This project explores financial data from the pharmaceutical industry, focusing on key performance indicators (KPIs) such as revenue, profit, and product-related metrics. The analysis begins with **Exploratory Data Analysis (EDA)** to understand trends, distributions, and relationships between KPIs, followed by **ARIMA forecasting**. Finally, the project transitions to a **Random Forest regression model** to improve forecasting accuracy.

## Key Features:
- **Exploratory Data Analysis (EDA)** on pharmaceutical KPIs
- **ARIMA Forecasting** for financial metrics (revenue, profit, etc.)
- Transition from ARIMA to **Random Forest** for improved prediction accuracy
- Model Tuning and Hyperparameter Optimization
- Model Evaluation using **MSE** (Mean Squared Error) and comparison of forecasted vs. actual values
- Data Visualizations for Analysis and Forecasting

## Tools Used:
- Python
- Pandas
- Scikit-Learn
- Statsmodels (ARIMA)
- Matplotlib
- Seaborn

## Project Overview:
### Step 1: Exploratory Data Analysis (EDA)
The project begins with an in-depth exploration of financial KPIs related to pharmaceutical products such as **vaccines**, **OTC drugs**, and **prescription medications**. EDA helps identify key trends and relationships in the data, as well as patterns in product sales, profit margins, and revenue variance. Visualizations and statistical summaries were used to reveal underlying trends.

### Step 2: ARIMA Forecasting
The ARIMA model was initially chosen to predict future financial metrics. However, ARIMA's limitations in capturing non-linear relationships and seasonal effects led to the exploration of a more complex model.

### Step 3: Transition to Random Forest
Due to the ARIMA model's limitations in handling the complexity of the financial data, **Random Forest** was implemented to improve accuracy. After hyperparameter tuning, the Random Forest model outperformed the ARIMA model in terms of Mean Squared Error (MSE):
- **ARIMA MSE**: 14,072,554,065.84
- **Random Forest MSE**: 12,670,440,477.92 (after hyperparameter tuning)

This improvement highlights the benefits of ensemble learning methods like Random Forest in handling complex data with multiple variables.

### Getting Started:
1. Clone this repository: `git clone https://github.com/yourusername/pharmaceutical-financial-analysis.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis and forecasting script: `python pharmaceutical_financial_analysis.py`

For a detailed walkthrough of the project, check out the Jupyter notebooks and Python scripts in this repository.


