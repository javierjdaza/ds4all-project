import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu
import glob
import os

def student_12():

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
    def make_prediction(df_input:pd.DataFrame):
        models = load_models('12')
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
                st.success(f'{materia}  \n  With probability of {round(predict_proba*100,3)}%')

        if len(resultados_prediccion_bad) > 0:
            st.write('The following subjects the student will have "Saber 11" Score below than 60 ')
            for i in resultados_prediccion_bad.to_dict(orient = 'records'):
                materia = i['materia']
                if materia == 'Ingles':
                    predict_proba = i['predict_proba']
                    st.warning(f'{materia}  \n  With probability of aim Score Above 80 of {round(predict_proba*100,3)}%')
                else:
                    predict_proba = i['predict_proba']
                    st.warning(f'{materia}  \n  With probability of aim Score Above 60 of {round(predict_proba*100,3)}%')


        # st.dataframe(resultados_prediccion)
        # if predict == 0:
        #     text_result = f'According to the features inputed, the student will have a Saber 12 Score below 340.  \n  The probability of the student for overcome the 340 score was: {predict_proba}%'
        #     st.error(text_result)
        # elif predict ==1:
        #     text_result = f'According to the features inputed, the student will have a Saber 12 Score above 340 with a probability of: {predict_proba}%'
        #     st.success(text_result)



    def create_column_feature_drill(name :str):
        min_value = 0
        max_value = 100
        name_display = name.capitalize()
        name_display = name_display.replace('_',' ').replace('sim','- Drill')
        
        slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = 60)
        return slider

    def create_column_feature_psat(name :str):
        min_value = 0
        max_value = 800
        name_display = name.capitalize()
        name_display = name_display.replace('_',' ').replace('psat','- Psat')
        
        slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = 60)
        return slider

    def create_column_feature_grades_12(name :str):
        min_value = 0 # int(df[name].min())
        max_value = 100 # int(df[name].max())
        name_display = name.capitalize()
        name_display = name_display.replace('_',' ').replace('notas 12','- 12 Grades').replace('nio','ño')
        
        slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = 60 )
        return slider

    def create_column_feature_grades_12(name :str):
        min_value = 0 # int(df[name].min())
        max_value = 100 # int(df[name].max())
        name_display = name.capitalize()
        name_display = name_display.replace('_',' ').replace('notas 12','- 12 Grades').replace('nio','ño')
        
        slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = 60)
        return slider



    def put_graph(file_path:str, title:str):
        st.subheader(title)
        st.markdown('#')
        st.image(file_path)



    st.subheader('Setup the values for make the prediction')
    st.write('---')
    st.subheader('Drill Results - 12📈')
    c1,c2,c3,c4,c5 = st.columns(5)
    c6,c7,c8,c9,c10= st.columns(5)
    st.write('---')

    # =======================
    # Simulacro
    # =======================
    with c1:
        biologia_sim_12 = create_column_feature_drill('biologia_sim_12')
    with c2:
        cts_sim_12 = create_column_feature_drill('cts_sim_12')
    with c3:
        competencias_ciudadanas_sim_12 = create_column_feature_drill('competencias_ciudadanas_sim_12')
    with c4:
        fisica_sim_12 = create_column_feature_drill('fisica_sim_12')
    with c5:
        ingles_sim_12 = create_column_feature_drill('ingles_sim_12')
    with c6:
        lectura_critica_sim_12 = create_column_feature_drill('lectura_critica_sim_12')
    with c7:
        matematicas__espec__sim = create_column_feature_drill('matematicas_(espec)_sim_12')
    with c8:
        matematicas__cuant__sim = create_column_feature_drill('matematicas_(cuant)_sim_12')
    with c9:
        quimica_sim_12 = create_column_feature_drill('quimica_sim_12')
    with c10:
        sociales_sim_12 = create_column_feature_drill('sociales_sim_12')

    # =======================
    # Grades 12
    # =======================
    st.subheader('Grades - 12 📙')
    g1,g2,g3,g4 = st.columns(4)
    g5,g6,g7,g8 = st.columns(4)
    g9,g10,g12,g12 = st.columns(4)
    st.write('---')
  
    with g1:
        economia_notas_12 = create_column_feature_grades_12('economia_notas_12')
    with g2:
        espaniol_notas_12 = create_column_feature_grades_12('espaniol_notas_12')
    with g3:
        filosofia_notas_12 = create_column_feature_grades_12('filosofia_notas_12')
    with g4:
        fisica_notas_12 = create_column_feature_grades_12('fisica_notas_12')
    with g5:
        matematicas_notas_12 = create_column_feature_grades_12('matematicas_notas_12')
    with g6:
        quimica_notas_12 = create_column_feature_grades_12('quimica_notas_12')
    with g7:
        ingles_notas_12 = create_column_feature_grades_12('ingles_notas_12')
    with g8:
        disciplina_notas_12 = create_column_feature_grades_12('disciplina_notas_12')
    with g9:
        ciencias_sociales_notas_12 = create_column_feature_grades_12('ciencias_sociales_notas_12')
    with g10:
        estadistica_notas_12 = create_column_feature_grades_12('estadistica_notas_12')




    # Create the dataframe with all inputs
    dict_input = {

    'economia_notas_12': economia_notas_12,
    'espaniol_notas_12': espaniol_notas_12,
    'filosofia_notas_12': filosofia_notas_12,
    'espaniol_notas_12': espaniol_notas_12,
    'matematicas_notas_12': matematicas_notas_12,
    'fisica_notas_12' : fisica_notas_12,
    'quimica_notas_12': quimica_notas_12,
    'ingles_notas_12': ingles_notas_12,
    'matematicas_notas_12':matematicas_notas_12,
    'disciplina_notas_12':disciplina_notas_12,
    'biologia_sim_12' : biologia_sim_12,
    'cts_sim_12' : cts_sim_12,
    'competencias_ciudadanas_sim_12' : competencias_ciudadanas_sim_12,
    'fisica_sim_12' : fisica_sim_12,
    'ingles_sim_12' : ingles_sim_12,
    'lectura_critica_sim_12' : lectura_critica_sim_12,
    'matematicas_(espec)_sim_12' : matematicas__espec__sim,
    'matematicas_(cuant)_sim_12' : matematicas__cuant__sim,
    'quimica_sim_12' : quimica_sim_12,
    'sociales_sim_12' : sociales_sim_12,
    'ciencias_sociales_notas_12' : ciencias_sociales_notas_12,
    'estadistica_notas_12' : estadistica_notas_12
    }
 

    df_input = pd.DataFrame([dict_input])
    st.subheader('Table of your inputs: ')
    st.dataframe(df_input)
    st.write('---')

    # ===================
    # Button Prediction
    # ===================
    but1,but2,but3,but4,but5 = st.columns(5)

    with but3:
        prediction = st.button('Make Prediction🥷')
                
    if prediction:
        st.balloons()
        prediccion = make_prediction(df_input)
        
    
    