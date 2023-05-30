# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:05:13 2023

@author: fzbri
"""
import pandas as pd
import numpy as np
import time
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns

#from tensorflow import keras
#from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch


def xgb_grid_search(X_train, y_train):
    """
    Performs grid search to find the best hyperparameters for the XGBoost classifier.

    Args:
        X_train: The training features.
        y_train: The target values for the training set.

    Returns:
        dict: A dictionary containing the best hyperparameters found during grid search.
    """
    print("----------------------------------PERFORMING GRID SEARCH----------------------------------")
    
    # Define the hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0],
        #'colsample_bytree': [0.5, 0.7, 1.0],
    }
    
    # Create a XGBClassifier model
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc')
    
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy')
    
    # Perform GridSearch
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    
    # Print best parameters
    print(f"Best parameters: {best_params}")
    
    return best_params #{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7}



def perform_xgboost(X_train, X_test, y_train, y_test):
    """
    Trains the XGBoost classifier, performs cross-validation, evaluates performance metrics
    (ROC & Metrics & Confusion Matrix), and saves the model.

    Args:
        X_train: The training features.
        X_test: The testing features.
        y_train: The target values for the training set.
        y_test: The target values for the testing set.

    Returns:
        XGBClassifier: The trained XGBoost classifier model.
    """
    print("----------------------------------RUNNING RANDOM XGBOOST'S GRID SEARCH----------------------------------")
    # Commentented the GridSearch and put directly the best parameters into the Model to avoid running again
    # Perform GridSearch to get the best parameters
    #best_params = xgb_grid_search(X_train, y_train)
    
    print("----------------------------------RUNNING XGBOOST----------------------------------")
    # Start the timer
    start_time = time.time()

    # Train XGBClassifier model
    model_xgb = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=500, subsample=0.7) #**best_params

    # Cross-validation

    scores = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {scores}")
    print(f"Mean cross-validation ROC AUC score: {np.mean(scores)}")

    # Fit the model
    model_xgb.fit(X_train, y_train)

    # Predict on test set
    y_pred_xgb = model_xgb.predict(X_test)
    # Predict probabilities
    y_pred_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nElapsed Time: {elapsed_time:.2f} seconds")

    roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
    print(f'ROC AUC: {roc_auc_xgb:.4f}')
    
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f'Accuracy: {acc_xgb:.4f}')
    
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    #print(f'Confusion Matrix:\n{cm_xgb}')

    # Create dataframe with confusion matrix and labels
    cm_df_xgb = pd.DataFrame(cm_xgb, columns=['Predicted 0', 'Predicted 1'], index=['True 0', 'True 1'])
    print(f'\nConfusion Matrix:\n{cm_df_xgb}')
    
    # Calculate F1-score
    report_xgb = classification_report(y_test, y_pred_xgb)
    print(f'\nClassification Report:\n{report_xgb}')
    
    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgb)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_xgb)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.title('XGBoost ROC Curve')
    plt.plot(fpr, tpr, 'teal', label = 'ROC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # CM
    plt.figure(figsize=(7,5))
    sns.heatmap(cm_df_xgb, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # save model
    with open('models/model_xgb.pkl', 'wb') as f:
        pickle.dump(model_xgb, f)
        
    return model_xgb

        
#%% RF

def rf_grid_search(X_train, y_train):
    """
    Performs grid search to find the best hyperparameters for the Random Forest classifier.

    Args:
        X_train: The training features.
        y_train: The target values for the training set.

    Returns:
        tuple: A tuple containing the best model and best hyperparameters found during grid search.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params


def perform_random_forest(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates (ROC & Metrics & Confusion Matrix) the Random Forest classifier
    then saves it.

    Args:
        X_train: The training features.
        X_test: The testing features.
        y_train: The target values for the training set.
        y_test: The target values for the testing set.

    Returns:
        RandomForestClassifier: The trained Random Forest classifier model.
    """
    print("----------------------------------RUNNING RANDOM FOREST'S GRID SEARCH----------------------------------")
    # Commentented the GridSearch and put directly the best parameters into the Model to avoid running again
    # Perform grid search to find best parameters
    #best_model, best_params = rf_grid_search(X_train, y_train)
    #print("Best Parameters:", best_params)
    #print("Best Model:", best_model)
    print("----------------------------------RUNNING RANDOM FOREST----------------------------------")
    # Start the timer
    start_time = time.time()
    # Train RandomForestClassifier model
    model_rf = RandomForestClassifier(n_estimators=300, min_samples_split=2, random_state=42)
    #model_rf = RandomForestClassifier(**best_params, random_state=42) # Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
    
    # Perform cross-validation
    scores = cross_val_score(model_rf, X_train, y_train, cv=5)  # Change the number of folds as needed
    model_rf.fit(X_train, y_train)
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nElapsed Time: {elapsed_time:.2f} seconds")
    
    print("Cross-Validation Scores:", scores)
    print("Mean Cross-Validation Score:", scores.mean())
    print("Standard Deviation of Cross-Validation Scores:", scores.std())
    

    # Predict on test set and display accuracy
    y_pred_rf = model_rf.predict(X_test)
    y_pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]  # probabilities for the positive outcome
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f'Accuracy: {acc_rf:.4f}')

    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    #print(f'Confusion Matrix:\n{cm}')


    # Create dataframe with confusion matrix and labels
    cm_df_rf = pd.DataFrame(cm_rf, columns=['Predicted 0', 'Predicted 1'], index=['True 0', 'True 1'])
    print(f'\nConfusion Matrix:\n{cm_df_rf}')
    
    # Plot the confusion matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(cm_df_rf, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    
    # Calculate F1-score
    report_rf = classification_report(y_test, y_pred_rf)
    print(f'\nClassification Report:\n{report_rf}')
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rf)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.title('Random Forest ROC Curve')
    plt.plot(fpr, tpr, 'teal', label = 'ROC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # save model
    with open('models/model_rf.pkl', 'wb') as f:
        pickle.dump(model_rf, f)
        
    return model_rf

#%% Neural Network

""" Neural Network Hyperparameters
# Commented and bestparams directly put into the NN to avoid running the gridSearch (time-consuming)

def perform_nn(X_train, X_test, y_train, y_test):
"""
"""
    Trains and evaluates neural network models with different combinations of hyperparameters.

    Args:
        X_train: The training features.
        X_test: The testing features.
        y_train: The target values for the training set.
        y_test: The target values for the testing set.

    Returns:
        tuple: A tuple containing the best hyperparameters and the corresponding accuracy.
"""
"""
    print("----------------------------------RUNNING NEURAL NETWORKS----------------------------------")
    # Define the range of values for hyperparameters
    num_layers = [1, 2, 3]
    num_neurons = [16, 32, 64]
    dropout_rates = [0.25, 0.5, 0.75]

    # Initialize a dictionary to store accuracies for each combination of hyperparameters
    accuracies = {}

    # Iterate over hyperparameters and train models
    for nl in num_layers:
        for nn in num_neurons:
            for dr in dropout_rates:
                # Define the model architecture
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(nn, activation='relu', input_shape=(X_train.shape[1],)))
                model.add(tf.keras.layers.Dropout(dr))
                for i in range(nl - 1):
                    model.add(tf.keras.layers.Dense(nn, activation='relu'))
                    model.add(tf.keras.layers.Dropout(dr))
                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                # Compile the model
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                # Train the model
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                # Evaluate the model on the test set
                y_pred = model.predict(X_test)
                y_pred = (y_pred > 0.5).astype(int)
                acc = accuracy_score(y_test, y_pred)
                # Store the accuracy in the dictionary
                key = (nl, nn, dr)
                accuracies[key] = acc
                print(f'Layers: {nl}, Neurons: {nn}, Dropout: {dr}, Accuracy: {acc:.4f}')
                
    # Find the combination with the highest accuracy
    best_params = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_params]
    print(f'Best parameters: {best_params}, Accuracy: {best_acc:.4f}')
    """


#Best parameters: (2, 64, 0.25)
def perform_nn(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates (ROC & Metrics & Confusion Matrix) of the neural network model
    then saves.

    Args:
        X_train: The training features.
        X_test: The testing features.
        y_train: The target values for the training set.
        y_test: The target values for the testing set.

    Returns:
        Sequential: The trained neural network model.
    """
    print("----------------------------------RUNNING NEURAL NETWORKS----------------------------------")
    # Define the neural network architecture
    model_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.25),
        #tf.keras.layers.Dense(32, activation='relu'),
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the neural network
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Train the neural network
    history = model_nn.fit(X_train, y_train, epochs=42, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the performance of the neural network on the testing data
    test_loss, test_acc = model_nn.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc) # 0.921


    # Predict on test set and display accuracy
    y_pred_nn = model_nn.predict(X_test)
    y_pred_classes = np.round(y_pred_nn)
    acc_nn = accuracy_score(y_test, y_pred_classes)
    print(f'Accuracy: {acc_nn:.4f}')

    cm_nn = confusion_matrix(y_test, y_pred_classes)
    print(f'Confusion Matrix:\n{cm_nn}')

    # Create dataframe with confusion matrix and labels
    cm_df_nn = pd.DataFrame(cm_nn, columns=['Predicted 0', 'Predicted 1'], index=['True 0', 'True 1'])
    print(f'\nConfusion Matrix:\n{cm_df_nn}')
    
    # Calculate F1-score
    report_nn = classification_report(y_test, y_pred_classes)
    print(report_nn)

    model_nn.summary()

    model_nn.save('models/model_nn')
    
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(7,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    
    # Plot the confusion matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(cm_df_nn, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_nn)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='teal', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Neural Network ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return model_nn

