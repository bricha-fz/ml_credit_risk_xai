# -*- coding: utf-8 -*-
"""
Created on Mon May  8 21:01:40 2023

@author: fzbri
"""
# specify path

path = '../ml_credit_risk_xai'
import os
os.chdir(path)
import pickle
import pandas as pd
from data_processing import get_missing_values, fill_missing_values, clean_extreme_values, oversample_data, split_scale_data
from models import perform_xgboost, perform_random_forest, perform_nn
from xai import *
from xai_eli5 import eli_explain
from xai_dalex import dalex_explain, dalex_predict
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


func_var = {
    "train_model": False,
    "use_trained_model": True
    }


# Import dataset
df = pd.read_csv('./data/credit_risk_dataset.csv')   

cols_with_nan = df.columns[df.isna().any()].tolist()


df_complete, df_nan, predictors = get_missing_values(df)
fill_missing_values(df, df_complete, df_nan, predictors, cols_with_nan[0])
fill_missing_values(df, df_complete, df_nan, predictors, cols_with_nan[1])

# Delete extreme non -logical values
df = clean_extreme_values(df)

# oversample data
df =  oversample_data(df)


# Hot encoding:
# we have person_home_ownership, 
# Define a list of categorical column names
cat_cols = ['person_home_ownership', 'cb_person_default_on_file', 'loan_intent', 'loan_grade']

# Convert the categorical columns into one-hot encoding
df = pd.get_dummies(df, columns=cat_cols)

# split & scale
X, X_train, X_test, y_train, y_test, scaler = split_scale_data(df)


##### TRAIN MODELS
if func_var["train_model"]:
    model_xgb = perform_xgboost(X_train, X_test, y_train, y_test)
    model_rf = perform_random_forest(X_train, X_test, y_train, y_test)
    model_nn = perform_nn(X_train, X_test, y_train, y_test)


### OPEN TRAIN MODELS
if func_var["use_trained_model"]:
    with open('models/model_rf.pkl', 'rb') as f:
        model_rf = pickle.load(f)
    with open('models/model_xgb.pkl', 'rb') as f:
        model_xgb = pickle.load(f)
    model_nn = tf.keras.models.load_model("models/model_nn")


###### XAI
col_names = X.columns 

#%% SHAP EXPLAINERS:
    
# RF
explainer_rf, shap_values_rf, explanation_rf = explain_tree(model_rf, X_train, X_test[:2000], col_names, 'rf')
plot_shap_rf(explainer_rf, shap_values_rf, X_test[:50], col_names, explanation_rf, model_rf, row=1)

# XGB
explainer_xgb, shap_values_xgb, explanation_xgb = explain_tree(model_xgb, X_train, X_test, col_names, 'xgb')
plot_shap_xgb(explainer_xgb, shap_values_xgb, X_test, col_names, explanation_xgb, model_xgb, row=0)

# NN
explainer_nn, shap_values_nn = explain_nn(model_nn, X_train, X_test, col_names)
plot_shap_nn(explainer_nn, shap_values_nn, X_test, col_names, model_nn, row=0)



#%% DALEX EXPLAINERS:

# RF
dalex_explainer_rf = dalex_explain(model_rf, X_test, y_test, 'rf')
dalex_predict(dalex_explainer_rf, X_test[0], 'rf')

# XGB
dalex_explainer_xgb = dalex_explain(model_xgb, X_test, y_test, 'xgb')
dalex_predict(dalex_explainer_xgb, X_test[0], 'xgb')

# NN
dalex_explainer_nn = dalex_explain(model_nn, X_test, y_test, 'nn')
dalex_predict(dalex_explainer_nn, X_test[0], 'nn')


#%% ELI5 EXPLAINER

eli_explain(model_rf, X_test[:1], col_names.to_list(), 'rf')
eli_explain(model_xgb, X_test[:1], col_names.to_list(), 'xgb')

