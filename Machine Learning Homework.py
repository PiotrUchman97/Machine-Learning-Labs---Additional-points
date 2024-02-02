import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer  
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ARDL
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from models import ModelTrainer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Stata dataset into a pandas DataFrame
dataset = pd.read_stata(r'C:\Users\piotr\Downloads\tax_avoidance.dta')

# Select target variable and features
y = dataset['etr']
X = dataset[['ta', 'txt', 'pi', 'str', 'xrd', 'ni', 'ppent', 'intant', 'dlc', 'dltt',
             'capex', 'revenue', 'cce', 'adv', 'diff', 'roa', 'lev', 'intan', 'rd',
             'ppe', 'sale', 'cash_holdings', 'adv_expenditure', 'capex2_scaled',
             'capex2', 'cfc', 'dta', 'capex2_scaled', 'roa1', 'capex1', 'diff1', 'diff2', 'diff3']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an imputer instance
imputer = SimpleImputer(strategy='mean') 

# Fit and transform the imputer on the training data
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data using the same imputer
X_test_imputed = imputer.transform(X_test)

# Create a RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model with imputed data
rf_model.fit(X_train_imputed, y_train)

# Make predictions on the imputed test set
predictions_imputed = rf_model.predict(X_test_imputed)

# Evaluate the model with imputed data
mse_imputed = mean_squared_error(y_test, predictions_imputed)
rmse_imputed = mean_squared_error(y_test, predictions_imputed, squared=False)

print(f'Random Forest MSE: {mse_imputed}')
print(f'Random Forest RMSE: {rmse_imputed}')

# Visualization of Random Forest predictions
comp_rf = pd.DataFrame({'y_pred': predictions_imputed, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_rf, kind="reg", truncate=False)
plt.show()


#Naive model
naive_model = DummyRegressor(strategy='mean')
naive_model.fit(X_train_imputed, y_train)
predictions_naive_imputed = naive_model.predict(X_test_imputed)

# Evaluate the Naive model
mse_naive_imputed = mean_squared_error(y_test, predictions_naive_imputed)
rmse_naive_imputed = mean_squared_error(y_test, predictions_naive_imputed, squared=False)

print(f'Naive Model MSE: {mse_naive_imputed}')
print(f'Naive Model RMSE: {rmse_naive_imputed}')

# Visualization of Naive Model predictions
comp_naive = pd.DataFrame({'y_pred': predictions_naive_imputed, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_naive, kind="reg", truncate=False)
plt.show()

# Create a DecisionTreeRegressor model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree model with imputed data
dt_model.fit(X_train_imputed, y_train)

# Make predictions on the imputed test set
predictions_dt_imputed = dt_model.predict(X_test_imputed)

# Evaluate the Decision Tree model with imputed data
mse_dt_imputed = mean_squared_error(y_test, predictions_dt_imputed)
rmse_dt_imputed = mean_squared_error(y_test, predictions_dt_imputed, squared=False)

print(f'Decision Tree MSE: {mse_dt_imputed}')
print(f'Decision Tree RMSE: {rmse_dt_imputed}')

# Visualization of Decision Tree Model predictions
comp_dt = pd.DataFrame({'y_pred': predictions_dt_imputed, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_dt, kind="reg", truncate=False)
plt.show()

# Create an ElasticNet model
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) 

# Train the ElasticNet model with imputed data
elasticnet_model.fit(X_train_imputed, y_train)

# Make predictions on the imputed test set
predictions_elasticnet_imputed = elasticnet_model.predict(X_test_imputed)

# Evaluate the ElasticNet model with imputed data
mse_elasticnet_imputed = mean_squared_error(y_test, predictions_elasticnet_imputed)
rmse_elasticnet_imputed = mean_squared_error(y_test, predictions_elasticnet_imputed, squared=False)

print(f'ElasticNet MSE: {mse_elasticnet_imputed}')
print(f'ElasticNet RMSE: {rmse_elasticnet_imputed}')

# Visualization of ElasticNet Model predictions
comp_elasticnet = pd.DataFrame({'y_pred': predictions_elasticnet_imputed, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_elasticnet, kind="reg", truncate=False)
plt.show()

# Create a KNeighborsRegressor model
knn_model = KNeighborsRegressor(n_neighbors=5)  

# Train the KNN model with imputed data
knn_model.fit(X_train_imputed, y_train)

# Make predictions on the imputed test set
predictions_knn_imputed = knn_model.predict(X_test_imputed)

# Evaluate the KNN model with imputed data
mse_knn_imputed = mean_squared_error(y_test, predictions_knn_imputed)
rmse_knn_imputed = mean_squared_error(y_test, predictions_knn_imputed, squared=False)

print(f'KNN MSE: {mse_knn_imputed}')
print(f'KNN RMSE: {rmse_knn_imputed}')

## Visualisation of KNN
comp_knn = pd.DataFrame({'y_pred': predictions_knn_imputed, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_knn, kind="reg", truncate=False)
plt.show()

# Scale the features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create an SVR model
svm_model = SVR(kernel='linear', C=1.0)  # You can experiment with different kernel functions and C values

# Train the SVR model with scaled data
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
predictions_svm_scaled = svm_model.predict(X_test_scaled)

# Evaluate the SVR model with scaled data
mse_svm_scaled = mean_squared_error(y_test, predictions_svm_scaled)
rmse_svm_scaled = mean_squared_error(y_test, predictions_svm_scaled, squared=False)

print(f'SVM MSE: {mse_svm_scaled}')
print(f'SVM RMSE: {rmse_svm_scaled}')

# Visualization of SVR Model predictions
comp_svm = pd.DataFrame({'y_pred': predictions_svm_scaled, 'y_true': y_test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_svm, kind="reg", truncate=False)
plt.show()

# Assuming 'etr' is your time series variable
time_series = dataset['etr']

# Handling missing values
imputer = SimpleImputer(strategy='mean')
time_series_imputed = imputer.fit_transform(time_series.values.reshape(-1, 1)).ravel()

# Split the time series into training and testing sets
train_size = int(len(time_series_imputed) * 0.8)
train, test = time_series_imputed[:train_size], time_series_imputed[train_size:]

# Fit ARIMA model
order = (5, 1, 2) 
model = ARIMA(train, order=order)
fit_model = model.fit()

# Make predictions
predictions = fit_model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

# Evaluate the model
mse_arma = mean_squared_error(test, predictions)
rmse_arma = mean_squared_error(test, predictions, squared=False)

print(f'ARIMA MSE: {mse_arma}')
print(f'ARIMA RMSE: {rmse_arma}')

# Visualization of ARIMA Model predictions
comp_arma = pd.DataFrame({'y_pred': predictions, 'y_true': test})

# Jointplot with regression line
plt.figure(figsize=(10, 5))
sns.jointplot(x='y_pred', y='y_true', data=comp_arma, kind="reg", truncate=False)
plt.show()

# Random Forest Hyperparameter Tuning
rf_params = {
    'n_estimators': [50],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf_model, param_grid=rf_params, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
rf_grid.fit(X_train_imputed, y_train)

best_rf_model = rf_grid.best_estimator_

# Decision Tree Hyperparameter Tuning
dt_params = {
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

dt_model = DecisionTreeRegressor(random_state=42)
dt_grid = GridSearchCV(dt_model, param_grid=dt_params, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
dt_grid.fit(X_train_imputed, y_train)

best_dt_model = dt_grid.best_estimator_

# ElasticNet Hyperparameter Tuning
elasticnet_params = {
    'alpha': [0.1],
    'l1_ratio': [0.1],
    'max_iter': [1000]
}

elasticnet_model = ElasticNet(random_state=42)
elasticnet_grid = GridSearchCV(elasticnet_model, param_grid=elasticnet_params, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
elasticnet_grid.fit(X_train_imputed, y_train)

best_elasticnet_model = elasticnet_grid.best_estimator_

# Save local champions to a pickle file
local_champions = {
    'RandomForest': best_rf_model,
    'DecisionTree': best_dt_model,
    'ElasticNet': best_elasticnet_model
}

with open('local_champions.pkl', 'wb') as f:
    pickle.dump(local_champions, f)

# Print the best hyperparameters for each model
print("Best Random Forest Hyperparameters:", rf_grid.best_params_)
print("Best Decision Tree Hyperparameters:", dt_grid.best_params_)
print("Best ElasticNet Hyperparameters:", elasticnet_grid.best_params_)

mse_rf_imputed = mean_squared_error(y_test, predictions_imputed)

# Example DataFrame:
model_scores = pd.DataFrame({
    'Model': ['Random Forest', 'Naive Model', 'Decision Tree', 'ElasticNet', 'KNN', 'SVR', 'ARIMA'],
    'MSE': [mse_imputed, mse_naive_imputed, mse_dt_imputed, mse_elasticnet_imputed, mse_knn_imputed, mse_svm_scaled, mse_arma],
    'RMSE': [rmse_imputed, rmse_naive_imputed, rmse_dt_imputed, rmse_elasticnet_imputed, rmse_knn_imputed, rmse_svm_scaled, rmse_arma]
})

# Set ggplot style
plt.style.use('ggplot')

# Plotting
fig, ax = plt.subplots(figsize=(15, 7))
model_scores.plot(kind='bar', x='Model', y=['MSE', 'RMSE'], ax=ax, title="Model Comparison on TEST set", rot=0, fontsize=12)
ax.legend(fontsize=13)
ax.set_xlabel("MODELS", fontsize=13)
ax.set_ylabel("Error", fontsize=13)
plt.show()



### Randon Forest have to lowers MSE and RMSE so it is the best model