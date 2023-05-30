# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:16:48 2023

@author: fzbri
"""

import dalex as dx
import pandas as pd

#matplotlib.use('Qt5Agg')  # Specify the backend you want to use
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio

def dalex_explain(model, X, y, name):
    """
    Generates explanations for the models using the DALEX library and saves the plots as HTML files.
    
    Args:
        model: The model to be explained.
        X: The input features.
        y: The target variable.
        name: The name of the model.
    
    Returns:
        explainer: The Explainer object.
    """
    if name == 'rf':
        label = 'Random Forest'
    elif name == 'xgb':
        label = 'XGBoost'
    elif name == 'nn':
        label = 'Neural Network'
        
    X = convert_df(X)
    #test = convert_df(test)
    explainer = dx.Explainer(model, X, y, label=label)
    
    # Variable importance
    fig = explainer.model_parts().plot(show=False)

    # Display the plot using plotly's offline mode
    pyo.plot(fig, filename=f'results_xai/dalex/model_parts_plot_{name}.html', auto_open=True)
    
    # Model Profile
    model_profile = explainer.model_profile()
    fig2 = explainer.model_profile().plot(show=False)
    pyo.plot(fig2, filename=f'results_xai/dalex/model_profile_plot_{name}.html', auto_open=True)
    
    #Model diagnostic
    model_diagnostics = explainer.model_diagnostics().result
    
    # model perfomance
    model_perf = explainer.model_performance().result
    print(model_perf)
    fig3 = explainer.model_performance(model_type='classification').plot(geom='roc') #roc, ecdf, lift 
    pyo.plot(fig3, filename=f'results_xai/dalex/model_performance_plot_{name}.html', auto_open=True)
    
    return explainer
    

    
def dalex_predict(explainer, obs, name):
    """
    Generates prediction explanations for a specific observation using the DALEX library and saves the plots as HTML files.
    
    Args:
        explainer: The Explainer object.
        obs: The specific observation to explain.
        name: The name of the model.
    
    Returns:
        fig: The prediction explanation plot.
    """
    #explainer.predict_parts(obs).result
    fig=explainer.predict_parts(obs).plot(show=False)
    pyo.plot(fig, filename=f'results_xai/dalex/obs_{name}.html', auto_open=True) 
    fig2 = explainer.predict_parts(obs, type='shap').plot(show=False)
    #fig = explainer.predict_parts(obs, type='break_down').plot()
    pyo.plot(fig2, filename=f'results_xai/dalex/obs_shap_{name}.html', auto_open=True) 
    fig3 = explainer.predict_profile(obs).plot(show=False)
    pyo.plot(fig3, filename=f'results_xai/dalex/obs_profile_{name}.html', auto_open=True) 
    
    return fig
    
    
    
def convert_df(data):
    """
    Converts a numpy array into a pandas DataFrame with specific column names.
    
    Args:
        data: The numpy array to be converted.
    
    Returns:
        df: The pandas DataFrame with the converted data.
    """
    column_names = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
                'person_home_ownership_OWN', 'person_home_ownership_RENT',
                'cb_person_default_on_file_N', 'cb_person_default_on_file_Y',
                'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_A',
                'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
                'loan_grade_F', 'loan_grade_G']
    
    # Convert numpy array to DataFrame
    df = pd.DataFrame(data, columns=column_names)
    return df