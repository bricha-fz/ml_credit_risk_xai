# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:04:15 2023

@author: fzbri
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_missing_values(df):
    """
    Returns the complete and missing value subsets of a DataFrame, 
    along with the predictors used for linear imputation.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing three elements:
            - df_complete (DataFrame): Subset of the input DataFrame with complete rows (no missing values).
            - df_nan (DataFrame): Subset of the input DataFrame with rows containing missing values.
            - predictors (list): List of column names representing the predictors for the linear imputation.
    """
    df_complete = df.dropna()
    df_nan = df[df.isna().any(axis=1)]

    # select the predictors for the regression model
    predictors = ['person_age', 'person_income', 'loan_amnt', 'loan_status']
    
    return df_complete, df_nan, predictors


def fill_missing_values(df, df_complete, df_nan, predictors, col):
    """
    Fills missing values in a DataFrame column using regression-based imputation.

    Args:
        df (DataFrame): The original DataFrame.
        df_complete (DataFrame): Subset of the original DataFrame with complete rows (no missing values).
        df_nan (DataFrame): Subset of the original DataFrame with rows containing missing values.
        predictors (list): List of column names representing the predictors for the regression model.
        col (str): The name of the column to fill missing values for.

    Returns:
        None: This function modifies the input DataFrame in-place by filling missing values in the specified column.
    """
    # train the regression model using cross-validation
    regression_model = LinearRegression()
    #scores = cross_val_score(regression_model, df_complete[predictors], df_complete[col], cv=5)

    # fit the regression model on the complete observations
    regression_model.fit(df_complete[predictors], df_complete[col])

    # predict the missing values using the trained model
    df_nan[col+'_predicted'] = regression_model.predict(df_nan[predictors])

    # replace the missing values with the predicted values
    df[col].fillna(value=df_nan[col+'_predicted'], inplace=True)

def clean_extreme_values(df):
    """
    Cleans extreme values in specific columns of a DataFrame by setting them to NaN and removing rows with missing values.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A cleaned version of the input DataFrame with extreme values set to NaN and rows with missing values dropped.
    """
    # set the value of person_emp_length to NaN for rows where it equals 123
    df.loc[df['person_emp_length'] == 123, 'person_emp_length'] = np.nan
    # set the value of person_emp_length to NaN for rows where it equals 123
    df.loc[df['person_age'] == 144, 'person_age'] = np.nan
    df.loc[df['person_age'] == 123, 'person_age'] = np.nan
    df = df.dropna()
    return df



def oversample_data(df):
    """
    Performs oversampling on the df to balance the class distribution.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The oversampled DataFrame with balanced class distribution.
    """
    # Separate majority and minority classes
    df_majority = df[df.loan_status==0]
    df_minority = df[df.loan_status==1]
    
    # Oversample minority class
    df_minority_oversampled = resample(df_minority, 
                                       replace=True, # sample with replacement
                                       n_samples=len(df_majority), # match number in majority class
                                       random_state=42) # for reproducibility
    
    # Combine majority class with oversampled minority class
    df_oversampled = pd.concat([df_majority, df_minority_oversampled])
    
    # Check class distribution
    df_oversampled.loan_status.value_counts()
    return df_oversampled

def split_scale_data(df):
    """
    Splits the dataset into training and testing sets, and scales the data using StandardScaler.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing six elements:
            - X (DataFrame): The features of the original DataFrame without the target column.
            - X_train (ndarray): The scaled training features.
            - X_test (ndarray): The scaled testing features.
            - y_train (Series): The target values for the training set.
            - y_test (Series): The target values for the testing set.
            - scaler (StandardScaler): The fitted StandardScaler object used for scaling the data.
    """
    # Split the dataset into training and testing sets
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X, X_train, X_test, y_train, y_test, scaler






