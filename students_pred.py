import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu
import glob
import os

def student():
    
    def load_models(grade:str)->list:

        models = glob.glob('./models/*')
        models_ = [ i for i in models if os.path.basename(i).startswith(grade)]
        return models_

    # st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
    hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # Setup the model
    # model = joblib.load('./model/model.joblib')

    # Help Fuctions
    def make_prediction(df_input:pd.DataFrame,grade = str):
        models = load_models(grade)
        list_results = []
        for i in models:
            model = joblib.load(i)
            name_model = os.path.basename(i).split('_')[1]
            predict = model.predict(df_input)[0]
            predict_proba = model.predict_proba(df_input)[:,1][0]

            dict_results = {f'materia':f'{name_model}','predict':predict,'predict_proba':predict_proba}
            list_results.append(dict_results)

        resultados_prediccion = pd.DataFrame(list_results)
        resultados_prediccion['materia'] = resultados_prediccion['materia'].apply(lambda x: x.replace('SOCIALES','SOCIALES Y CIUDADANAS').capitalize())
        resultados_prediccion['materia'] = resultados_prediccion['materia'].apply(lambda x: x.replace('Math','MATEMATICAS').capitalize())
        resultados_prediccion['materia'] = resultados_prediccion['materia'].apply(lambda x: x.replace('Lect','LECTURA CRITICA ').capitalize())
        resultados_prediccion['materia'] = resultados_prediccion['materia'].apply(lambda x: x.replace('SOCIALES','SOCIALES Y CIUDADANAS').capitalize())

        # FILTRADO BUENOS Y MALOS
        resultados_prediccion_good = resultados_prediccion[resultados_prediccion['predict'] == 1]
        resultados_prediccion_bad = resultados_prediccion[resultados_prediccion['predict'] == 0]
        if len(resultados_prediccion_good) > 0:
            st.write('The following subjects the student will achieve score more than 60 in the "Saber Pro"')
            for i in resultados_prediccion_good.to_dict(orient = 'records'):
                materia = i['materia']
                predict_proba = i['predict_proba']
                st.success(f'{materia}  \n  With probability of {round(predict_proba,3)}%')

        if len(resultados_prediccion_bad) > 0:
            st.write('The following subjects the student will have "Saber 11" Score below than 60 ')
            for i in resultados_prediccion_bad.to_dict(orient = 'records'):
                materia = i['materia']
                if materia == 'Ingles':
                    predict_proba = i['predict_proba']
                    st.warning(f'{materia}  \n  With probability of aim Score Above 80 of {round(predict_proba,3)}%')
                else:
                    predict_proba = i['predict_proba']
                    st.warning(f'{materia}  \n  With probability of aim Score Above 60 of {round(predict_proba,3)}%')


        # st.dataframe(resultados_prediccion)
        # if predict == 0:
        #     text_result = f'According to the features inputed, the student will have a Saber 12 Score below 340.  \n  The probability of the student for overcome the 340 score was: {predict_proba}%'
        #     st.error(text_result)
        # elif predict ==1:
        #     text_result = f'According to the features inputed, the student will have a Saber 12 Score above 340 with a probability of: {predict_proba}%'
        #     st.success(text_result)
    st.subheader('Input Student information 👨‍🎓')

    col1,col2,col3,col4 = st.columns(4)
    st.write('---')
    
    with col2:
        student_code = st.text_input('Enter Student Code', placeholder = '20042003',value = '20042003')
    
    with col3:
        year = st.text_input('Enter Student Year', placeholder = '11',value = '11')
  


    # year_10_dataframes_filepaths = [ i for i in glob.glob('./data/transformed/*') if i.startswith('df_10')]
    # year_11_dataframes_filepaths = [ i for i in glob.glob('./data/transformed/*') if i.startswith('df_11')]
    # year_12_dataframes_filepaths = [ i for i in glob.glob('./data/transformed/*') if i.startswith('df_12')]

    if year == '10':
        df_input = pd.read_csv('./data/transformed/df_10_CIENCIAS.csv')
        del df_input['ciencias_saber_11']
        
    if year == '11':
        df_input = pd.read_csv('./data/transformed/df_11_CIENCIAS.csv')
        del df_input['ciencias_saber_11']
        
    if year == '12':
        df_input = pd.read_csv('./data/transformed/df_12_CIENCIAS.csv')
        del df_input['ciencias_saber_11']

    df_input = df_input[(df_input['codigo'] == int(student_code))]
    if len(df_input) > 0:
        del df_input['codigo']
        st.subheader('Student features')
        st.dataframe(df_input)
        st.write('---')
        df_polar = df_input.copy()
        df_polar.fillna(0,inplace=True)
        st.write('Scores of the Student Selected')
        p1,p2,p3,p4,p5,p6 = st.columns((1,1,1,4,1,1))
        st.write('---')
        with p3:  
            for i in df_polar.columns:
                df_polar[i] = df_polar[i].apply(lambda x: int(x))
            
            df_polar = df_polar.T.reset_index()
            df_polar.columns = ['theta','r']
            # st.dataframe(df_polar)
            fig = px.line_polar(df_polar, r='r', theta='theta', line_close=False,  width=750, height=750)
            fig.update_traces(fill='toself')

            st.plotly_chart(fig, use_container_width=False)
    else:
        st.error('Please enter a valid <<student code>> & <<year>>')

    
    
    # ===================
    # Button Prediction
    # ===================
    but1,but2,but3,but4,but5 = st.columns(5)
    with but3:
        prediction = st.button('Make Prediction🥷')
                
    if prediction:
        st.balloons()
        prediccion = make_prediction(df_input,year)
        
    
    