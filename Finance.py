#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


df1 = pd.read_csv('Finance_updated.csv')

# Selecting the first 24 rows (months)
df = df1.iloc[:24]

df.head()


# In[4]:


df.describe()


# ## Converting Months from " 2021-04-30 " format to "Month Year" format

# In[5]:


df['Month'] = pd.to_datetime(df['Month'])

# Format the 'month' column as "Month Year"
df['Month'] = df['Month'].dt.strftime('%B %Y')

# Display the DataFrame
print(df[['Month']])


# In[39]:


df.head()


# # Variance Analysis

# ## Revenue Variance Analysis

# In[40]:


#A negative revenue variance percentage means that the actual revenue for a given period (month) is less than the budgeted or forecasted revenue. 

df['Revenue_Variance_%'] = (df['Revenue_Variance']/df['Budgeted_Revenue'])*100
#print("Percentage of Revenue Variance : \n",df['Revenue_Variance_%'])

plt.figure(figsize=(10,6))
plt.plot(df['Revenue_Variance_%'], marker='o', linestyle='-', color='blue')
plt.title('Revenue Variance Percentage over Time')
plt.xlabel('Month')
plt.ylabel('Variance(%)')
plt.grid(True)
plt.show()


# ## Cost Variance Analysis

# In[41]:


df['Cost_Variance_%']=(df['Cost_Variance']/df['Budgeted_Cost_of_Production'])*100

plt.figure(figure=(10,6))
plt.plot(df['Cost_Variance_%'], marker='o', linestyle='-', color='red')
plt.title('Cost Variance Percentage (%)')
plt.ylabel('Variance(%)')
plt.xlabel('Month Year')
plt.grid(True)
plt.show()


# # Risk vs Profit Analysis

# In[22]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Risk_Score', y='Net_Profit', data=df, hue='Credit_Score', palette='coolwarm')
plt.title('Risk Score vs Net Profit')
plt.xlabel('Risk Score')
plt.ylabel('Net Profit')
plt.grid(True)
plt.show()

# Correlation between Risk and Net Profit
correlation_risk_profit = df['Risk_Score'].corr(df['Net_Profit'])
print(f"Correlation between Risk Score and Net Profit: {correlation_risk_profit}")


# # Profitability Analysis

# In[23]:


# Profitability over time
plt.figure(figsize=(10, 6))
plt.plot(df['Profit_Margin'], label='Profit Margin', marker='o', color='green')
plt.plot(df['Operating_Margin'], label='Operating Margin', marker='o', color='purple')
plt.title('Profitability Analysis (Profit Margin vs. Operating Margin)')
plt.xlabel('Months')
plt.ylabel('Margin (%)')
plt.legend()
plt.grid(True)
plt.show()


# # Time-Series Analysis

# ## Revenue and Profit Over Time

# In[24]:


plt.figure(figsize=(12, 6))
plt.plot(df['Revenue'], label='Revenue', marker='o', color='blue')
plt.plot(df['Net_Profit'], label='Net Profit', marker='o', color='orange')
plt.title('Revenue and Net Profit Over Time')
plt.xlabel('Months')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.show()


# # Growth Rate Analysis

# In[25]:


# Revenue Growth Rate Over Time
plt.figure(figsize=(10, 6))
plt.plot(df['Revenue_Growth_Rate'], label='Revenue Growth Rate', marker='o', color='purple')
plt.title('Revenue Growth Rate Over Time')
plt.xlabel('Months')
plt.ylabel('Growth Rate (%)')
plt.grid(True)
plt.show()


# # Forecasting Analysis

# ## ARIMA forecasting

# In[7]:


from statsmodels.tsa.arima.model import ARIMA

#setting month as index for model
df['Month'] = pd.to_datetime(df['Month'], format='%B %Y')
df.set_index('Month', inplace=True)

#plotting time series
plt.figure(figsize=(10,6))
plt.plot(df['Revenue'], label='Revenue')
plt.plot(df['Cost_of_Production'], label='Cost of Production')
plt.plot(df['Net_Profit'], label='Net Profit')
plt.title('Revenue, Cost of Production, and Net Proft Over Time')
plt.legend()
plt.show()


# ## Forecasting Revenue with ARIMA

# In[74]:


# Forecasting Revenue

model_revenue = ARIMA(df2['Revenue'], order=(1,1,1))
revenue_fit = model_revenue.fit()

#Forecast next 12 months
forecast_revenue = revenue_fit.forecast(steps=12)

#Plotting forecasted revenue
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Revenue'], label='Actual Revenue')
plt.plot(pd.date_range(start=df.index[-1], periods=13, freq='MS')[1:], forecast_revenue, label='Forecasted Revenue', linestyle='--')
plt.title('Revenue Forecast')
plt.legend()
plt.show()


# In[12]:


from sklearn.metrics import mean_absolute_error

# Fit ARIMA model on the first 500 entries (same as Random Forest)
model_revenue_arima = ARIMA(df2['Revenue'], order=(1,1,1))
revenue_fit_arima = model_revenue_arima.fit()

# Forecast the first 48 entries (instead of the next 12 months)
forecast_arima = revenue_fit_arima.predict(start=0, end=47)  # Predicting first 48 entries

# Now let's plot the ARIMA predictions against the actual values for the first 48 entries
actual_values_arima = df2['Revenue'][:48]  # Actual values for the first 48 entries

mae_arima = mean_absolute_error(actual_values_arima, forecast_arima)
print(f"Mean Absolute Error (MAE) for ARIMA: {mae_arima}")

plt.figure(figsize=(10,6))
plt.plot(range(48), actual_values_arima, marker='o', linestyle='-', color='blue', label='Actual (ARIMA)')
plt.plot(range(48), forecast_arima, marker='o', linestyle='--', color='green', label='Predicted (ARIMA)')
plt.title('Actual vs Predicted Revenue (ARIMA - First 48 Entries)')
plt.xlabel('Months')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()


# ## Forecasting Cost of Production with ARIMA

# In[73]:


# Forecasting Cost of Production

model_productioncost = ARIMA(df2['Cost_of_Production'], order=(1,1,1))
productioncost_fit = model_productioncost.fit()

#Forecast next 12 months
forecast_productioncost = productioncost_fit.forecast(steps=12)

#Plotting forecasted revenue
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Cost_of_Production'], label='Actual Cost of Production')
plt.plot(pd.date_range(start=df.index[-1], periods=13, freq='MS')[1:], forecast_productioncost, label='Forecasted Cost of Production', linestyle='--')
plt.title('Cost of Production Forecast')
plt.legend()
plt.show()


# ## Forecasting Net Profit with ARIMA

# In[45]:


# Forecasting Net Profit

model_profit = ARIMA(df['Net_Profit'], order=(1,1,1))
profit_fit = model_profit.fit()

#Forecast next 12 months
forecast_profit = profit_fit.forecast(steps=24)

#Plotting forecasted revenue
plt.figure(figsize=(10,6))
plt.plot(df.index, df['Net_Profit'], label='Actual Profit')
plt.plot(pd.date_range(start=df.index[-1], periods=25, freq='MS')[1:], forecast_profit, label='Forecasted Profit', linestyle='--')
plt.title('Profit Forecast')
plt.legend()
plt.show()


# ## Forecasting Revenue with Random Forest

# In[9]:


df2 = df1.iloc[:500]


# In[15]:


#Import libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Select features and target variable ( we want to predict 'Revenue')
X = df2[['Cost_of_Production','Operating_Expenses']]  # Features
y = df2['Revenue']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
rf_model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
print(f"Mean Squared Error: {mse}")

# Step 8: Plot actual vs predicted revenue
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Revenue')
plt.plot(y_pred, label='Predicted Revenue', linestyle='--')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Index')
plt.ylabel('Revenue')
plt.legend()
plt.show()


# In[47]:


df1.describe()


# ## Comparing Baseline predictions(mean) MSE to Random Forest MSE 

# In[16]:


# Assuming 'y_test' is your actual Revenue values (ground truth)
# and 'rf_predictions' are the predictions from your Random Forest model

# Step 1: Calculate the mean of the actual values (Revenue)
mean_revenue = np.mean(y_test)

# Step 2: Create baseline predictions where every prediction is the mean
baseline_predictions = np.full_like(y_test, mean_revenue)

# Step 3: Calculate the MSE for the baseline model
baseline_mse = mean_squared_error(y_test, baseline_predictions)

# Step 4: Print and compare MSEs
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))
print("Baseline MSE (mean prediction):", baseline_mse)


# ## Fine tuning Hperparamenters 

# In[17]:


from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestRegressor(random_state=42)

# Specify the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
}

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Get the best model from the grid search
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred_tuned = best_rf_model.predict(X_test)

# Calculate the MSE for the tuned model
from sklearn.metrics import mean_squared_error
mse_tuned = mean_squared_error(y_test, y_pred_tuned)

print(f"Random Forest MSE after tuning: {mse_tuned}")


#  ### Plotting Predicted vs Actual Values of Fine tuned Model

# In[18]:


#Plot Actual vs. Predicted Values
y_pred_fine_tuned = best_rf_model.predict(X_test)

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Revenue', marker='o', linestyle='-', color='blue')
plt.plot(y_pred_fine_tuned, label='Predicted Values', marker='x', linestyle='--', color='orange')
plt.title('Actual vs Predicted Values for Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()


# In[19]:


# Select the first 48 entries for plotting

df_subset = df2[:48] # First 48 entries

y_pred_subset = y_pred_fine_tuned[:48]
actual_values_subset = y_train[:48]

mae_random_forest = mean_absolute_error(actual_values_subset, y_pred_subset)
print(f"Mean Absolute Error (MAE) for Random Forest: {mae_random_forest}")

# Plotting the actual vs predicted values for the first 48 entries
plt.figure(figsize=(10,6))
plt.plot(range(48), actual_values_subset, marker='o', linestyle='-', color='blue', label='Actual')
plt.plot(range(48), y_pred_subset, marker='o', linestyle='--', color='red', label='Predicted')
plt.title('Actual vs Predicted Revenue (First 48 Entries)')
plt.xlabel('Months')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()


# In[20]:


#Plotting Actual vs Predicted for both ARIMA and Random Forest
plt.figure(figsize=(10, 6))
plt.plot(range(48), actual_values_arima, marker='o', linestyle='-', color='blue', label='Actual')
plt.plot(range(48), forecast_arima, marker='o', linestyle='--', color='green', label='Predicted (ARIMA)')
plt.plot(range(48), y_pred_subset, marker='o', linestyle='--', color='red', label='Predicted (Random Forest)')
plt.title('Actual vs Predicted Revenue (ARIMA and Random Forest - First 48 Entries)')
plt.xlabel('Months')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




