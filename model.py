import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Data Processing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

# Get variable names
# from varname import nameof

# Pipeline diagram
from sklearn import set_config

# MODELS
# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, roc_auc_score
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

def evaluation(y_test, y_predict, title = 'Confusion Matrix'):
    cm = confusion_matrix(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    accuracy = accuracy_score(y_test,y_predict)
    f1 = f1_score(y_test,y_predict)
    roc = roc_auc_score(y_test, y_predict)
    metrics = {'Accuracy':accuracy,
                'precision':precision,
                'Recall':recall,
                'f1':f1,
                'roc' : roc
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
    
    st.pyplot(plt.show())
    return metrics_df



def plot_feature_importances(model,X_train):
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    # customized number 
    num_features = 10 

    plt.figure(figsize=(10,10))
    plt.title('Feature Importances')

    # only plot the customized number of features
    plt.barh(range(num_features), importances[indices[-num_features:]], color='black', align='center')
    plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
    plt.xlabel('Relative Importance')
    st.pyplot(plt.show())

def predict_with_results():

    df_10_MATH  = pd.read_csv('./data/transformed/df_10_MATH.csv')
    df_10_LECT = pd.read_csv('./data/transformed/df_10_LECT.csv')
    df_10_INGLES = pd.read_csv('./data/transformed/df_10_INGLES.csv')
    df_10_CIENCIAS = pd.read_csv('./data/transformed/df_10_CIENCIAS.csv')
    df_10_SOCIALES = pd.read_csv('./data/transformed/df_10_SOCIALES.csv')
    df_11_MATH = pd.read_csv('./data/transformed/df_11_MATH.csv')
    df_11_LECT = pd.read_csv('./data/transformed/df_11_LECT.csv')
    df_11_INGLES = pd.read_csv('./data/transformed/df_11_INGLES.csv')
    df_11_CIENCIAS = pd.read_csv('./data/transformed/df_11_CIENCIAS.csv')
    df_11_SOCIALES = pd.read_csv('./data/transformed/df_11_SOCIALES.csv')
    df_12_MATH = pd.read_csv('./data/transformed/df_12_MATH.csv')
    df_12_LECT = pd.read_csv('./data/transformed/df_12_LECT.csv')
    df_12_INGLES = pd.read_csv('./data/transformed/df_12_INGLES.csv')
    df_12_CIENCIAS = pd.read_csv('./data/transformed/df_12_CIENCIAS.csv')
    df_12_SOCIALES = pd.read_csv('./data/transformed/df_12_SOCIALES.csv')

    dataframes = [df_10_MATH,df_10_LECT,df_10_INGLES,df_10_CIENCIAS,df_10_SOCIALES,
    df_11_MATH,df_11_LECT,df_11_INGLES,df_11_CIENCIAS,df_11_SOCIALES,
    df_12_MATH,df_12_LECT,df_12_INGLES,df_12_CIENCIAS,df_12_SOCIALES]




    datasets_models_dict = {
        'df_10_MATH': 'rf_classifier',
        'df_10_LECT': 'rf_classifier',
        'df_10_INGLES': 'rf_classifier',
        'df_10_CIENCIAS' : 'gb_classifier',
        'df_10_SOCIALES' : 'gb_classifier',
        'df_11_MATH': 'rf_classifier',
        'df_11_LECT' : 'rf_classifier',
        'df_11_INGLES' : 'rf_classifier',
        'df_11_CIENCIAS' : 'gb_classifier',
        'df_11_SOCIALES': 'rf_classifier',
        'df_12_MATH' : 'rf_classifier',
        'df_12_LECT' : 'rf_classifier',
        'df_12_INGLES' : 'rf_classifier',
        'df_12_CIENCIAS' : 'rf_classifier',
        'df_12_SOCIALES' : 'gb_classifier',}

    contador = 0
    logs_metrics = pd.DataFrame()
    for i in dataframes:
        try:
            del i['codigo']
        except:
            pass
        X = i.drop(columns = i.columns[-1], axis = 1)
        y = i[f'{i.columns[-1]}']
        X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


        model_choose = datasets_models_dict[list(datasets_models_dict.keys())[contador]]
        dataset_name = list(datasets_models_dict.keys())[contador]



        if model_choose == 'rf_classifier':

            model = RandomForestClassifier()
    
        
        elif model_choose == 'gb_classifier':

            model = GradientBoostingClassifier()
            

        # Build the pipeline
        
    
        # Build the pipeline
        pipeline = Pipeline([
                        ('Model', model)
                    ])
        
        pipeline.fit(X_train, y_train);
        y_predict = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:,1]
        # print(list(datasets_models_dict.keys())[contador])
        col1,col2,col3 = st.columns(3)
        with col1:
            logs_metrics_ = evaluation(y_test, y_predict)
            logs_metrics_['data_Set'] = list(datasets_models_dict.keys())[contador]
            logs_metrics_['model'] = model_choose
        with col2:
            st.dataframe(logs_metrics_)
        with col3:
            plot_feature_importances(model,X_train)
        print('-'*50)
        contador +=1
