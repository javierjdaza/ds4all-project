from functools import cache
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Data Processing
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer



# Pipeline diagram
from sklearn import set_config

# MODELS
# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, roc_auc_score


def evaluation(y_test, y_predict, title = 'Confusion Matrix'):
    cm = confusion_matrix(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    accuracy = accuracy_score(y_test,y_predict)
    f1 = f1_score(y_test,y_predict)
    metrics = {'Accuracy':accuracy,
                'precision':precision,
                'Recall':recall,
                'f1':f1
                 }

    metrics_df = pd.DataFrame([metrics])
    # print('Recall: ', recall)
    # print('Accuracy: ', accuracy)
    # print('Precision: ', precision)
    # print('F1: ', f1)
    # display(metrics_df)
    sns.heatmap(cm,  cmap= 'Blues', annot=True, fmt='g', annot_kws=    {'size':20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)
    plt.show();

@st.cache
def load_dataset()->pd.DataFrame:

    df = pd.read_csv('./data/marymount_dataset_transformed.csv')
    return df


def train(df: pd.DataFrame, model_name:str, percentaje_test:float):

    X = df.drop(columns = {'target'}, axis = 1)
    y = df['target']

    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = percentaje_test, random_state = 0)


    numerical_columns = X_train.select_dtypes(exclude='O').columns.to_list()


    num_pipe = Pipeline([

        ('num_preprocessing',SimpleImputer(strategy='mean')),
        ('Standard Scaler', StandardScaler()) # rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.
        ])

    preprocessor = ColumnTransformer([
        ('Numeric Features', num_pipe, numerical_columns)
        ])

    if model_name == 'Random Forest':
        model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='auto',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100,
                            n_jobs=-1, oob_score=False, random_state=23, verbose=0,
                            warm_start=False)

    elif model_name == 'Logistic Regression':
        model = LogisticRegression()
    
    elif model_name == 'K Neighbors Classifier':
        model = KNeighborsClassifier()


    # Build the pipeline
    pipe = Pipeline([
                        ('preprocessor',preprocessor),
                        # ('Smote', SMOTE(random_state=0)),
                        ('Model', model)
                    ])
    pipe.fit(X_train, y_train);
    y_predict = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:,1]

    
    return 