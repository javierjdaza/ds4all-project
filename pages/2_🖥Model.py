import streamlit as st
import pandas as pd
from model import predict_with_results
import os
st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed", use_column_width=True)
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.cache()
def put_graphs_metrics(datasets_strings,dict_datos):
    for i in datasets_strings:
        for j in dict_datos:
            if list(j.keys())[0] == i:
                st.subheader(list(j.keys())[0])
                st.markdown('#')
                g1,g2,g3,g4 = st.columns(4)
                cp1,cp2,cp3 = st.columns(3)
                st.markdown('#')
                
                sep1,sep2,sep3 = st.columns(3)
                with sep2:
                    st.write('➖'*20)
                t1,t2 = st.columns((5,1))
                
                c1,c2,c3,c4,c5 = st.columns(5)
                st.write('---')
                with g2:
                    cm = j[i][0]
                    st.image( cm, use_column_width=False)
                with cp2:
                    st.caption('Confusion Matrix')
                with t1:
                    st.markdown('#')
                    fi = j[i][1]
                    st.image( os.path.join(os.getcwd(), fi), use_column_width=True)
                with c3:
                    st.caption('Feature Importance')
                # with g3:
                #     metrics = j[i][2]
                #     st.image( os.path.join(os.getcwd(), metrics), use_column_width=True)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png', use_column_width=True)
with col4:
    st.image('./images/correlation.png', use_column_width=True)
st.write('----')


st.title('Model 🖥')
st.write('---')
st.markdown('#')

st.subheader('Here below we will drill down all the steps that allow us to build the solution.📌')
st.write('We include the raw code used to come up the solution thats perfom better according to the MaryMount request.')
st.write('---')
with st.expander('Code Part 🧩'):
    st.subheader('Importing neccesary libraries')
    st.code('''

    # Files handling
    import os
    import glob
    import warnings
    warnings.filterwarnings('ignore')

    # Data Visualization
    import matplotlib.image as pl, use_column_width=Truet
    from matplotlib import colors
    import seaborn as sns

    # Data Processing
    import pandas as pd
    pd.set_option('display.max_columns', None)
    import numpy as np
    from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler
    from sklearn.compose import ColumnTransformer
    from imblearn.pipeline import Pipeline

    # Pipeline diagram
    from sklearn import set_config

    # MODELS
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, roc_auc_score

    # Save serializable python objects
    import joblib

    ''')
    st.write('---')
    st.markdown('#')
    st.subheader('Reading the input DataSets')
    st.code('''
    ausencias = pd.read_csv('./data/raw/Ausencias.csv')
    definitivas = pd.read_csv('./data/raw/Definitivas_asignaturas.csv')
    estudiantes = pd.read_csv('./data/raw/Listado_estudiantes.csv')
    psat = pd.read_csv('./data/raw/Pruebas_PSAT.csv')
    saber_11 = pd.read_csv('./data/raw/Pruebas_Saber_11.csv')
    simulacro = pd.read_csv('./data/raw/Simulacro_pruebas_saber_11.csv')

    ''')

    st.write('---')
    st.markdown('#')
    st.subheader('Data transformation & Wrangling')
    st.code('''
    # SABER 11 DATA WRANGLING
    saber_11 = saber_11.pivot_table(index=['codigo','anio_escolar'], columns=['asignatura'],values='resultado').reset_index()
    saber_11 = saber_11.rename_axis(None, axis=1)
    saber_11.columns = [f'{j.lower().replace("","ni").replace(" ","_").strip()}_saber_11'  for j in saber_11.columns ]
    saber_11.rename(columns={f'codigo_saber_11':'codigo'}, inplace= True)
    del saber_11['anio_escolar_saber_11']
    del saber_11['global_saber_11']
    saber_11_materias = [ j for j in saber_11.columns if j.endswith('saber_11') ]
    for i in saber_11_materias:
        if i == 'ingles_saber_11':
            saber_11[i] = saber_11[i].apply(lambda x: 1 if x >=80 else 0)
        else:
            saber_11[i] = saber_11[i].apply(lambda x: 1 if x >=60 else 0)

    # PSAT DATA WRANGLING
    psat = psat.pivot_table(index=['codigo','anio_escolar'], columns=['asignatura'],values='resultado').reset_index()
    psat.drop_duplicates(subset= 'codigo', keep= 'first', inplace = True)
    psat = psat.rename_axis(None, axis=1)
    psat.columns = [f'{j.lower().replace(" ","_").strip()}_psat'  for j in psat.columns ]
    psat.rename(columns={f'codigo_psat':'codigo'}, inplace= True)
    del psat['anio_escolar_psat']
    del psat['combinado_psat']

    # GRADES DATA WRANGLING
    definitivas_10 = definitivas[definitivas.grado == 10]
    definitivas_10 = definitivas_10.groupby(['codigo','asignatura'])['resultado'].mean().to_frame().reset_index()
    definitivas_10 = definitivas_10.pivot_table(index=['codigo'], columns=['asignatura'],values='resultado').reset_index()
    definitivas_10 = definitivas_10.rename_axis(None, axis=1)
    definitivas_10.columns = [f'{j.lower().replace(" ","_").replace("","ni").strip()}_notas_10'  for j in definitivas_10.columns ]
    definitivas_10.rename(columns={f'codigo_notas_10':'codigo'}, inplace= True)

    definitivas_11 = definitivas[definitivas.grado == 11]
    definitivas_11 = definitivas_11.groupby(['codigo','asignatura'])['resultado'].mean().to_frame().reset_index()
    definitivas_11 = definitivas_11.pivot_table(index=['codigo'], columns=['asignatura'],values='resultado').reset_index()
    definitivas_11 = definitivas_11.rename_axis(None, axis=1)
    definitivas_11.columns = [f'{j.lower().replace(" ","_").replace("","ni").strip()}_notas_11'  for j in definitivas_11.columns ]
    definitivas_11.rename(columns={f'codigo_notas_11':'codigo'}, inplace= True)

    definitivas_12 = definitivas[definitivas.grado == 12]
    definitivas_12 = definitivas_12.groupby(['codigo','asignatura'])['resultado'].mean().to_frame().reset_index()
    definitivas_12 = definitivas_12.pivot_table(index=['codigo'], columns=['asignatura'],values='resultado').reset_index()
    definitivas_12 = definitivas_12.rename_axis(None, axis=1)
    definitivas_12.columns = [f'{j.lower().replace(" ","_").replace("","ni").strip()}_notas_12'  for j in definitivas_12.columns ]
    definitivas_12.rename(columns={f'codigo_notas_12':'codigo'}, inplace= True)


    # DRILL DATA WRANGLING
    simulacro['resultado'] = simulacro['resultado'].apply(lambda x: str(x).strip().replace(',','.'))
    simulacro['resultado'] = simulacro['resultado'].apply(lambda x: float(x))


    simulacro_11 = simulacro[simulacro.grado == 11]
    simulacro_11 = simulacro_11.groupby(['codigo','asignatura'])['resultado'].mean().to_frame().reset_index()
    simulacro_11 = simulacro_11.pivot_table(index=['codigo'], columns=['asignatura'],values='resultado').reset_index()
    simulacro_11 = simulacro_11.rename_axis(None, axis=1)
    simulacro_11.columns = [f'{j.lower().replace(" ","_").replace("","ni").strip()}_sim_11'  for j in simulacro_11.columns ]
    simulacro_11.rename(columns={f'codigo_sim_11':'codigo'}, inplace= True)
    del simulacro_11['def_sim_11']

    simulacro_12 = simulacro[simulacro.grado == 12]
    simulacro_12 = simulacro_12.groupby(['codigo','asignatura'])['resultado'].mean().to_frame().reset_index()
    simulacro_12 = simulacro_12.pivot_table(index=['codigo'], columns=['asignatura'],values='resultado').reset_index()
    simulacro_12 = simulacro_12.rename_axis(None, axis=1)
    simulacro_12.columns = [f'{j.lower().replace(" ","_").replace("","ni").strip()}_sim_12'  for j in simulacro_12.columns ]
    simulacro_12.rename(columns={f'codigo_sim_12':'codigo'}, inplace= True)
    del simulacro_12['def_sim_12']
    ''')

    st.write('---')
    st.markdown('#')
    st.subheader('Merging All the data Grouped by Year')
    st.code('''
    df_10 = definitivas_10.merge(saber_11, on = 'codigo', how = 'inner')
    df_10 = df_10.merge(psat, on = 'codigo', how = 'inner')

    df_11 = definitivas_11.merge(saber_11, on = 'codigo', how = 'inner')
    df_11 = df_11.merge(simulacro_11, on = 'codigo', how = 'inner')
    del df_11['ciencias_sociales_notas_11']
    df_11.dropna(inplace=True)

    df_12 = definitivas_12.merge(saber_11, on = 'codigo', how = 'inner')
    df_12 = df_12.merge(simulacro_12, on = 'codigo', how = 'inner')
    df_12.dropna(inplace=True)
    ''')


    st.write('---')
    st.markdown('#')
    st.subheader('Auxiliar Fuction for extract the target columns by Grade and Year')
    st.code('''
    def keep_materia_saber(df:pd.DataFrame,materia:str)->pd.DataFrame:
        """Get Rid all the saber 11 values, and just keep the "materia" result of the parameter

        Args:
            df (pd.DataFrame): Dataframe with all the columns (all saber 11 scores)
            materia (str): the name of the target

        Returns:
            (pd.DataFrame): the dataframe transformed with target value corresponding to the "materia" saber 11
        """
        col_add = df[[materia]].copy()
        columns = [ i for i in df.columns if not i.endswith('saber_11') ]
        df = df[columns]
        df[materia] = col_add

        return df

    math_column = 'matematicas_saber_11'
    lectura_column = 'lectura_critica_saber_11'
    ingles_column = 'ingles_saber_11'
    ciencias_column = 'ciencias_saber_11'
    sociales_column = 'sociales_y_ciudadanas_saber_11'
    ''')


    st.write('---')
    st.markdown('#')
    st.subheader('Applying the fuction keep_materia_saber()')
    st.code('''
    # Matematicas

    df_10_MATH = keep_materia_saber(df_10,math_column)
    df_11_MATH = keep_materia_saber(df_11,math_column)
    df_12_MATH = keep_materia_saber(df_12,math_column)

    # lectura
    df_10_LECT = keep_materia_saber(df_10,lectura_column)
    df_11_LECT = keep_materia_saber(df_11,lectura_column)
    df_12_LECT = keep_materia_saber(df_12,lectura_column)

    # Ingles
    df_10_INGLES= keep_materia_saber(df_10,ingles_column)
    df_11_INGLES= keep_materia_saber(df_11,ingles_column)
    df_12_INGLES= keep_materia_saber(df_12,ingles_column)

    # ciencias_column
    df_10_CIENCIAS = keep_materia_saber(df_10,ciencias_column)
    df_11_CIENCIAS = keep_materia_saber(df_11,ciencias_column)
    df_12_CIENCIAS = keep_materia_saber(df_12,ciencias_column)

    # sociales_column
    df_10_SOCIALES = keep_materia_saber(df_10,sociales_column)
    df_11_SOCIALES = keep_materia_saber(df_11,sociales_column)
    df_12_SOCIALES = keep_materia_saber(df_12,sociales_column)
    ''')


    st.write('---')
    st.markdown('#')
    st.subheader('Fuctions for Visualization later')
    st.code('''
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
        display(metrics_df)
        sns.heatmap(cm,  cmap= 'Blues', annot=True, fmt='g', annot_kws=    {'size':20})
        plt.xlabel('predicted', fontsize=18)
        plt.ylabel('actual', fontsize=18)
        plt.title(title, fontsize=18)
        
        plt.show();
        return metrics_df



    def plot_feature_importances(model,X_train):
        features = X_train.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)

        #  number of top features to plot
        num_features = 10 

        plt.figure(figsize=(10,10))
        plt.title('Feature Importances')

        plt.barh(range(num_features), importances[indices[-num_features:]], color='black', align='center')
        plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
        plt.xlabel('Relative Importance')
        plt.show()
    ''')


    st.write('---')
    st.markdown('#')
    st.subheader('Packing all the datasets in a list for iterative process later')
    st.code('''

    dataframes = [df_10_MATH,df_10_LECT,df_10_INGLES,df_10_CIENCIAS,df_10_SOCIALES,
    df_11_MATH,df_11_LECT,df_11_INGLES,df_11_CIENCIAS,df_11_SOCIALES,
    df_12_MATH,df_12_LECT,df_12_INGLES,df_12_CIENCIAS,df_12_SOCIALES]

    ''')


    st.write('---')
    st.markdown('#')
    st.subheader('Model & Visualize the output metrics & feature importance')
    st.code('''
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

        # Split the datasets for testing purpouse (70% Train, 30% test)
        X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


        model_choose = datasets_models_dict[list(datasets_models_dict.keys())[contador]]
        dataset_name = list(datasets_models_dict.keys())[contador]


        # Picking the correct model for each dataset using the "datasets_models_dict" dictionary
        if model_choose == 'rf_classifier':
            model = RandomForestClassifier()
        
        elif model_choose == 'gb_classifier':
            model = GradientBoostingClassifier()
                
    
        # Build the pipeline
        pipeline = Pipeline([
                        ('Model', model)
                    ])
        
        # Train the model
        pipeline.fit(X_train, y_train);
        # Make the prediction
        y_predict = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:,1]

        # Plot the performance metrics
        logs_metrics_ = evaluation(y_test, y_predict)
        logs_metrics_['data_Set'] = list(datasets_models_dict.keys())[contador]
        logs_metrics_['model'] = model_choose
        logs_metrics = logs_metrics.append(logs_metrics_)
        dataset_name_joblib = dataset_name.replace('df_','')

        # Saving a copy of the serializable python object (model)
        joblib.dump(pipeline,f'./models/{dataset_name_joblib}_{model_choose}.joblib')

        # Plot Feature Importance
        plot_feature_importances(model,X_train)
        contador +=1

    ''')

with st.expander('Results 👨‍💻'):
    
    datasets_strings = [
    'df_10_MATH',
    'df_10_LECT',
    'df_10_INGLES',
    'df_10_CIENCIAS' ,
    'df_10_SOCIALES' ,
    'df_11_MATH',
    'df_11_LECT' ,
    'df_11_INGLES' ,
    'df_11_CIENCIAS' ,
    'df_11_SOCIALES',
    'df_12_MATH' ,
    'df_12_LECT' ,
    'df_12_INGLES' ,
    'df_12_CIENCIAS' ,
    'df_12_SOCIALES' ]


    dict_datos = [{'df_10_MATH': ['./graphs/metrics graphs/df_10_MATH_cm.png','./graphs/metrics graphs/df_10_MATH_fi.png']},
    {'df_10_LECT': ['./graphs/metrics graphs/df_10_LECT_cm.png','./graphs/metrics graphs/df_10_LECT_fi.png']},
    {'df_10_INGLES': ['./graphs/metrics graphs/df_10_INGLES_cm.png','./graphs/metrics graphs/df_10_INGLES_fi.png']},
    {'df_10_CIENCIAS': ['./graphs/metrics graphs/df_10_CIENCIAS_cm.png','./graphs/metrics graphs/df_10_CIENCIAS_fi.png']},
    {'df_10_SOCIALES': ['./graphs/metrics graphs/df_10_SOCIALES_cm.png','./graphs/metrics graphs/df_10_SOCIALES_fi.png']},
    {'df_11_MATH': ['./graphs/metrics graphs/df_11_MATH_cm.png','./graphs/metrics graphs/df_11_MATH_fi.png']},
    {'df_11_LECT': ['./graphs/metrics graphs/df_11_LECT_cm.png','./graphs/metrics graphs/df_11_LECT_fi.png']},
    {'df_11_INGLES': ['./graphs/metrics graphs/df_11_INGLES_cm.png','./graphs/metrics graphs/df_11_INGLES_fi.png']},
    {'df_11_CIENCIAS': ['./graphs/metrics graphs/df_11_CIENCIAS_cm.png','./graphs/metrics graphs/df_11_CIENCIAS_fi.png']},
    {'df_11_SOCIALES': ['./graphs/metrics graphs/df_11_SOCIALES_cm.png','./graphs/metrics graphs/df_11_SOCIALES_fi.png']},
    {'df_12_MATH': ['./graphs/metrics graphs/df_12_MATH_cm.png','./graphs/metrics graphs/df_12_MATH_fi.png']},
    {'df_12_LECT': ['./graphs/metrics graphs/df_12_LECT_cm.png','./graphs/metrics graphs/df_12_LECT_fi.png']},
    {'df_12_INGLES': ['./graphs/metrics graphs/df_12_INGLES_cm.png','./graphs/metrics graphs/df_12_INGLES_fi.png']},
    {'df_12_CIENCIAS': ['./graphs/metrics graphs/df_12_CIENCIAS_cm.png','./graphs/metrics graphs/df_12_CIENCIAS_fi.png']},
    {'df_12_SOCIALES': ['./graphs/metrics graphs/df_12_SOCIALES_cm.png','./graphs/metrics graphs/df_12_SOCIALES_fi.png']}]
    
    st.subheader('Matrix of confusion')
    st.write(' ')
    st.write('''On the x-axis you can find the value predicted by the model and on the y-axis the value of the real values of the data with which the model was trained, a good result is when the data is located on the diagonal because it implies that correctly predicted the actual value.''')
    st.subheader('Feature importance')
    st.write(' ')
    st.write('''This graph allows you to analyze for each grade and subject of knowing 11 which variables (subjects, PSAT test or subjects in drill test) are key to obtaining a good result in the subject of knowing 11. This information will allow teachers and administrators to take decisions at the grade level about what strategies to apply in the subjects presented in the graph to improve their results and as a consequence improve test results know 11''')
    st.write('---')

    put_graphs_metrics(datasets_strings,dict_datos)



with st.expander('Lets Play with models 🎲'):

    pass