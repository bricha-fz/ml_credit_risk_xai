# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:07:05 2023

@author: fzbri
"""

from eli5 import show_prediction, explain_weights




def eli_explain(model, X_test, col_names, name):
    """
    Generates explanations for a model using the ELI5 library and saves the explanations as HTML files.
    
    Args:
        model: The model to be explained.
        X_test: The testing data.
        col_names: The column names of the input features.
        name: The name of the model.
    
    Returns:
        explanation_data: The explanation data in HTML format.
    """
    # Obtain feature importances Return an explanation of estimator parameters (weights).
    explanation = explain_weights(model, feature_names=col_names)
    print(explanation)

    # Show predictions with explanations
    for i in range(len(X_test)):
        print(f"Prediction for instance {i+1}:")
        explanation = show_prediction(model, X_test[i], feature_names=col_names)
        html_representation = explanation.data
        with open(f'results_xai/eli5/eli5_{name}_{i}.html', "w") as f:
            f.write(html_representation)
        print("----------------------------------------")
    return explanation.data

