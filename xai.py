# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:40:40 2023

@author: fzbri
"""

import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np


def explain_nn(model, X_train, X_test, col_names):
    """
    Explains the predictions of a neural network model using SHAP values.
    
    Args:
        model: The neural network model to be explained.
        X_train: The training data used to create the explainer.
        X_test: The testing data for which SHAP values are generated.
        col_names: The column names of the input features.
    
    Returns:
        explainer: A DeepExplainer object used for generating SHAP values.
        shap_values: The SHAP values for the testing data.
    """
    # Create a DeepExplainer object
    explainer = shap.DeepExplainer(model, X_train)
    
    # Generate SHAP values for the testing data
    shap_values = explainer.shap_values(X_test)
    
    #plot_shap_nn(explainer, shap_values, X_test, col_names, model, row=0)
    # save_values
    with open('results_xai/shap_values_nn.pkl', 'wb') as f:
        pickle.dump(shap_values, f)
        
    return explainer, shap_values
        
#%% Tree explainer for both rf & xgb
        
def explain_tree(model, X_train, X_test, col_names, name):
    """
    Explains the predictions of a tree-based model (RF + XGBoost) using SHAP values and saves the results.
    
    Args:
        model: The tree-based model to be explained.
        X_train: The training data used to create the explainer.
        X_test: The testing data for which SHAP values are generated.
        col_names: The column names of the input features.
        name: The name of the tree-based model ("xgb" for XGBoost, other values for other tree-based models).
    
    Returns:
        explainer: A TreeExplainer object used for generating SHAP values.
        shap_values: The SHAP values for the testing data.
        explanation: The Explanation object containing the SHAP values, base values, data, and feature names.
    """
    explainer = shap.TreeExplainer(model)
    #explainer = shap.TreeExplainer(model, data= X_test, model_output='probability') #, feature_perturbation='interventional'
    
    shap_values = explainer.shap_values(X_test)
    
    # Wrap the SHAP values in an Explanation object
    #explanation = shap.Explanation(values=shap_values, base_values=model.base_score, data=X_test, feature_names=col_names)
    # Wrap the SHAP values in an Explanation object
    if name == "xgb":
        explanation = shap.Explanation(values=shap_values, base_values=model.base_score, data=X_test, feature_names=col_names)
        #PLOTS
        plot_shap_xgb(explainer, shap_values, X_test, col_names, explanation, model)
    else:
        explanation = shap.Explanation(values=shap_values, data=X_test, feature_names=col_names)
        plot_shap_rf(explainer, shap_values, X_test, col_names, explanation, model)

    
    
    # save_values
    with open(f'results_xai/shap_values_{name}.pkl', 'wb') as f:
        pickle.dump(shap_values, f)
    with open(f'results_xai/explanation_{name}.pkl', 'wb') as f:
        pickle.dump(explanation, f)
    with open(f'results_xai/explainer_{name}.pkl', 'wb') as f:
        pickle.dump(explainer, f)
        
    
    return explainer, shap_values, explanation

#%% PLOT NN

#explanation, model,
def plot_shap_nn(explainer, shap_values, X_test, col_names, model, row=1): 
    """
    Plots the SHAP summary plot and waterfall plot for a neural network model.
    
    Args:
        explainer: The DeepExplainer object used for generating SHAP values.
        shap_values: The SHAP values for the testing data.
        X_test: The testing data.
        col_names: The column names of the input features.
        model: The neural network model.
        row: The index of the row in the testing data to visualize (default is 1).
    
    Returns:
        None
    """
    model_prediction = model.predict(X_test[row].reshape(1, -1)) #
    print(model_prediction)
    # Generate the SHAP summary plot
    shap.summary_plot(shap_values[0], X_test, feature_names=col_names)
    shap.summary_plot(shap_values, X_test, feature_names=col_names)
    
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][row,:], X_test[row,:], feature_names=col_names) #, max_display=1
    #shap.force_plot(explainer.expected_value[0], shap_values[0][row,:]  ,X_test[row,:],feature_names=col_names)
    #shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)
    
#%%
def plot_shap_rf(explainer, shap_values, X_test, col_names, explanation, model, row=0):
    """
    Plots the SHAP summary plot, dependence plots, force plots, and waterfall plots for the random forest model.
    
    Args:
        explainer: The TreeExplainer object used for generating SHAP values.
        shap_values: The SHAP values for the testing data.
        X_test: The testing data.
        col_names: The column names of the input features.
        explanation: The Explanation object containing the SHAP values, base values, data, and feature names.
        model: The random forest model.
        row: The index of the row in the testing data to visualize (default is 0).
    
    Returns:
        None
    """
    # Generate the SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=col_names) #plot_type="violin" for single plot
    # summary plot for each specific class
    shap.summary_plot(shap_values[0], X_test, feature_names=col_names)
    shap.summary_plot(shap_values[1], X_test, feature_names=col_names)
    
    model_prediction = model.predict(X_test[row].reshape(1, -1)) 
    print('********************************', model_prediction[0])
    print('********************************', type(model_prediction))
    
    #RDF
    shap.dependence_plot(0, shap_values[0], X_test, feature_names=col_names)  # 'col_name' is the name of the feature

    if model_prediction[0] == int(0) :
        #print(model_prediction)
        shap.force_plot(explainer.expected_value[0], shap_values[1][row,:], X_test[row,:], feature_names=col_names, matplotlib=True)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1][row,:], feature_names=col_names)
        
    if model_prediction[0] == int(1) : 
        #print(model_prediction)
        shap.force_plot(explainer.expected_value[1], shap_values[1][row,:], X_test[row,:], feature_names=col_names, matplotlib=True)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][row,:], feature_names=col_names)


    

    
    
#%% SHAP XGB PLOTS 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_shap_xgb(explainer, shap_values, X_test, col_names, explanation, model, row=0):
    """
    Plots the SHAP summary plot, force plot, waterfall plot, and bar plot for the XGBoost model.
    
    Args:
        explainer: The DeepExplainer object used for generating SHAP values.
        shap_values: The SHAP values for the testing data.
        X_test: The testing data.
        col_names: The column names of the input features.
        explanation: The Explanation object containing the SHAP values, base values, data, and feature names.
        model: The XGBoost model.
        row: The index of the row in the testing data to visualize (default is 0).
    
    Returns:
        None
    """
    
    model_prediction = model.predict(X_test[row].reshape(1, -1)) #
    print('**********************', model_prediction)
    # Generate the SHAP summary plot
    
    shap.summary_plot(shap_values, X_test, feature_names=col_names)

    #row = np.array(explanation.data[row,:])
    #row_sum =np.sum(row)
    shap.force_plot(explainer.expected_value, shap_values[row,:], X_test[row,:], feature_names=col_names, matplotlib=True)

    base_value, shap_values_row = explainer.expected_value, shap_values[row,:]
    # Convert base_value (log-odds) to probability
    base_value_prob = sigmoid(base_value)
    
    # Convert prediction (log-odds) to probability
    prediction_prob = sigmoid(base_value + np.sum(shap_values_row))
    
    print("Base Value (Probability): ", base_value_prob)
    print("Prediction (Probability): ", prediction_prob)    
    

    # waterfall
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[row,:], X_test[row,:], feature_names=col_names)
    # Generate the bar plot
    shap.plots.bar(explanation, max_display=22)
    shap.plots.bar(explanation[row])

    #beeswarn
    shap.plots.beeswarm(explanation)
    shap.plots.heatmap(explanation[:100, :])

    #scatter plot plots per column for all
    shap.plots.scatter(explanation[:,row], color=explanation[:,1])
    
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0:100,:], X_test[0:100,:], feature_names=col_names)
    
    # Save the force plot as an HTML file
    force_plot_html = f"force_plot.html"
    shap.save_html(force_plot_html, force_plot)
    



