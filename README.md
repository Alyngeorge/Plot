1. Importing Required Libraries

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import joblib  # To save the model
from hyperopt import fmin, tpe, hp, Trials  # Import Hyperopt for TPE
from sklearn.linear_model import LinearRegression  # Import for stacking

    pandas (pd): Used for handling and manipulating datasets.
    numpy (np): Provides mathematical operations and array handling.
    XGBRegressor (from xgboost): An efficient gradient boosting algorithm for regression.
    RandomForestRegressor (from sklearn.ensemble): A machine learning model using an ensemble of decision trees.
    train_test_split: Splits the dataset into training and testing sets.
    cross_val_score: Performs cross-validation for model evaluation.
    mean_squared_error, mean_absolute_error, r2_score: Metrics for model performance evaluation.
    StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
    KNNImputer: Fills missing values using k-nearest neighbors.
    boxcox (from scipy.stats): A transformation to normalize skewed data.
    inv_boxcox: The inverse function to revert Box-Cox transformed data.
    joblib: Saves and loads models efficiently.
    hyperopt (fmin, tpe, hp, Trials): Used for hyperparameter tuning using the Tree-structured Parzen Estimator (TPE) optimization method.
    LinearRegression: A simple linear regression model, used in model stacking.

2. Defining Hyperparameter Distributions

Hyperparameter tuning is done using Hyperopt. We define different hyperparameter distributions for various parameters.
Precipitation Rate

param_dist_precipitation = {
    'n_estimators': hp.quniform('n_estimators', 500, 1200, 50),
    'max_depth': hp.choice('max_depth', [4, 5, 6, 7, 8, 9, 10, 11]),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 5),
    'max_bin': hp.quniform('max_bin', 20, 256, 5)
}

    Defines a set of possible values for the hyperparameters of the XGBRegressor model.
    hp.quniform means we sample integer values within the range.
    hp.choice selects one value from a list.
    hp.uniform samples a floating-point number from a range.

Similarly, hyperparameter distributions are defined for:

    Soil Moisture (param_dist_soil_moisture)
    Wind Speed (param_dist_windspeed)
    Surface Temperature (param_dist_surface_temp)
    Deep Soil Temperature (param_dist_deep_soil_temp)
    Random Forest (param_dist_rf)

3. Data Loading & Preprocessing

try:
    df = pd.read_csv('sorted_data.csv')
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise

    Tries to load sorted_data.csv using pandas.
    If the file cannot be read, an error message is printed.

4. Define Target Columns

columns_to_predict = [
    'PRECIPITATION RATE', 'SOIL MOISTURE', 'WIND SPEED',
    'SURFACE TEMPERATURE', 'DEEP SOIL TEMPERATURE'
]

    These are the columns we want to predict using machine learning models.

5. Feature Engineering

(Adding Temporal and Trigonometric Features)

df['SEASON'] = (df['MONTH'] % 12 + 3) // 3
df['SIN_MONTH'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['COS_MONTH'] = np.cos(2 * np.pi * df['MONTH'] / 12)

    SEASON: Converts the month into a season (Winter, Summer, etc.).
    SIN_MONTH and COS_MONTH: Converts month into cyclical features using sine and cosine.

6. Create Lag Features

def create_lag_features(df, column):
    df[f'{column}_LAG1'] = df[column].shift(1)
    df[f'{column}_LAG2'] = df[column].shift(2)

# Create lag features for each target column
for column in columns_to_predict:
    create_lag_features(df, column)

    Creates lag features, meaning past values of a column:
        LAG1: The value from the previous row (t-1).
        LAG2: The value from two rows before (t-2).
    Helps capture temporal dependencies in the data.

7. Create Rolling Mean Features

# Rolling Mean Features for all five parameters
for column in columns_to_predict:
    df[f'{column}_ROLLING_MEAN'] = df[column].rolling(window=3).mean()

    Creates a rolling mean (3-period moving average) for each target column.

8. Define Feature Columns

metadata_columns = ['YEAR', 'MONTH', 'AVRG ELEVATION', 'MIN ELEVATION', 'MAX ELEVATION']

engineered_features = [
    'SEASON', 'SIN_MONTH', 'COS_MONTH',
    'PRECIPITATION RATE_LAG1', 'PRECIPITATION RATE_LAG2', 'PRECIPITATION RATE_ROLLING_MEAN',
    'SOIL MOISTURE_LAG1', 'SOIL MOISTURE_LAG2', 'SOIL MOISTURE_ROLLING_MEAN',
    'WIND SPEED_LAG1', 'WIND SPEED_LAG2', 'WIND SPEED_ROLLING_MEAN',
    'SURFACE TEMPERATURE_LAG1', 'SURFACE TEMPERATURE_LAG2', 'SURFACE TEMPERATURE_ROLLING_MEAN',
    'DEEP SOIL TEMPERATURE_LAG1', 'DEEP SOIL TEMPERATURE_LAG2', 'DEEP SOIL TEMPERATURE_ROLLING_MEAN'
]

feature_columns = engineered_features + metadata_columns

    Defines metadata features (static variables like elevation).
    Defines engineered features (season, sine/cosine, lag features, rolling means).
    Combines them into final feature columns.

9. Handle Missing Values Using KNNImputer

imputer = KNNImputer(n_neighbors=5)
df[engine... 

    Uses KNN imputation to fill missing values in engineered features.
    n_neighbors=5: Uses the 5 nearest neighbors to predict missing values.

Summary

    Loads data from CSV.
    Defines target columns to predict.
    Creates engineered features (season, trigonometry, lag features, rolling mean).
    Uses KNNImputer to handle missing values.
    Prepares hyperparameter search space for models.
    Ready for model training and tuning.
