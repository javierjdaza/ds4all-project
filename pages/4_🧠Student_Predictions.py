import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Setup the model
model = joblib.load('./model/model.joblib')

# Help Fuctions
def make_prediction(df_input:pd.DataFrame,model:joblib):
    predict = model.predict(df_input)[0]
    predict_proba = model.predict_proba(df_input)[:,1][0]

    if predict == 0:
        text_result = f'According to the features inputed, the student will have a Saber 11 Score below 340.  \n  The probability of the student for overcome the 340 score was: {predict_proba}%'
        st.error(text_result)
    elif predict ==1:
        text_result = f'According to the features inputed, the student will have a Saber 11 Score above 340 with a probability of: {predict_proba}%'
        st.success(text_result)

def create_column_feature_drill(name):
    min_value = 0
    max_value = 100
    name_display = name.capitalize()
    name_display = name_display.replace('_',' ').replace('sim','- Drill')
    
    slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = int(df[name].mean()))
    return slider

def create_column_feature_psat(name):
    min_value = df[name].min()
    max_value = df[name].max()
    name_display = name.capitalize()
    name_display = name_display.replace('_',' ').replace('psat','- Psat')
    
    slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = int(df[name].mean()))
    return slider

def create_column_feature_grades_11(name):
    min_value = 0 # int(df[name].min())
    max_value = 100 # int(df[name].max())
    name_display = name.capitalize()
    name_display = name_display.replace('_',' ').replace('notas 11','- 11 Grades').replace('nio','ño')
    
    slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = int(df[name].mean()) )
    return slider

def create_column_feature_grades_12(name):
    min_value = 0 # int(df[name].min())
    max_value = 100 # int(df[name].max())
    name_display = name.capitalize()
    name_display = name_display.replace('_',' ').replace('notas 12','- 12 Grades').replace('nio','ño')
    
    slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = int(df[name].mean()))
    return slider

def create_column_feature_absences(name):
    min_value = 0 # int(df[name].min())
    max_value = 20 # int(df[name].max())
    name_display = name.capitalize()
    name_display = name_display.replace('_',' ').replace('ausencias 11','- 11 Absences').replace('ausencias 12','- 12 Absences').replace('nio','ño')
    
    slider = st.slider(name_display, min_value=min_value, max_value=max_value, value = int(df[name].mean()))
    return slider

def put_graph(file_path:str, title:str):
    st.subheader(title)
    st.markdown('#')
    st.image(file_path)

df = pd.read_csv('./data/marymount_dataset_transformed.csv')
del df['target']

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png')
with col4:
    st.image('./images/correlation.png')
st.write('----')



st.title('Student Predictions 🧠')
st.write('---')

student_expander = st.expander('Student Prediction')
new_student_expander = st.expander('New Student Prediction')



with student_expander:
    col1,col2,col3 = st.columns(3)
    st.write('---')
    col4,col5 = st.columns(2)
    
    # p1,p2 = st.columns(2)
    st.write('---')
    with col2:
        student_code = st.text_input('Enter Student Code', placeholder = '20032002',value = '20022028')

    # with col4:
    #     put_graph(file_path = './graphs/spider_graph.png', title = 'Spyder Chart')
    # with col5:
    #     put_graph(file_path = './graphs/kmeans_graph.jpeg', title = 'Segmentation Chart')
    # with col6:
    #     put_graph(file_path = './graphs/line_graph.png', title = 'Line Chart')
    if student_code in [str(i) for i in df['codigo'].to_list()]:
        df_polar = df.copy()
        df_polar.fillna(0,inplace=True)
        df_polar['codigo'] = df_polar['codigo'].apply(lambda x: str(x))
        df_polar = df_polar[df_polar['codigo'] == student_code]
        st.write('Scores of the Student Selected')
        st.dataframe(df_polar)
        p1,p2,p3,p4,p5,p6 = st.columns((1,1,1,4,1,1))
        with p3:  
            

            for i in df_polar.columns:
                df_polar[i] = df_polar[i].apply(lambda x: int(x))
            del df_polar['codigo']
            df_polar = df_polar.T.reset_index()
            df_polar.columns = ['theta','r']
            # st.dataframe(df_polar)
            fig = px.line_polar(df_polar, r='r', theta='theta', line_close=False,  width=750, height=750)
            fig.update_traces(fill='toself')

            st.plotly_chart(fig, use_container_width=False)
    else:
        st.warning('Please Select a valid Student Code')

with new_student_expander:
    st.subheader('Setup the values for make the prediction')
    st.write('---')
    st.subheader('Drill Results 📈')
    c1,c2,c3,c4,c5 = st.columns(5)
    c6,c7,c8,c9,c10= st.columns(5)
    st.write('---')

    # =======================
    # Simulacro
    # =======================
    with c1:
        biologia_sim = create_column_feature_drill('biologia_sim')
    with c2:
        cts_sim = create_column_feature_drill('cts_sim')
    with c3:
        competencias_ciudadanas_sim = create_column_feature_drill('competencias_ciudadanas_sim')
    with c4:
        fisica_sim = create_column_feature_drill('fisica_sim')
    with c5:
        ingles_sim = create_column_feature_drill('ingles_sim')
    with c6:
        lectura_critica_sim = create_column_feature_drill('lectura_critica_sim')
    with c7:
        matematicas__espec__sim = create_column_feature_drill('matematicas_(espec)_sim')
    with c8:
        matematicas__cuant__sim = create_column_feature_drill('matematicas_(cuant)_sim')
    with c9:
        quimica_sim = create_column_feature_drill('quimica_sim')
    with c10:
        sociales_sim = create_column_feature_drill('sociales_sim')

    # =======================
    # PSAT
    # =======================
    st.subheader('Psat Results ✍')
    c11,c12= st.columns(2)
    st.write('---')

    with c11:
        math_psat = create_column_feature_drill('math_psat')
    with c12:
        reading_and_writing_psat = create_column_feature_drill('reading_and_writing_psat')
    
    # =======================
    # Grades 11
    # =======================
    st.subheader('Grades 11 📙')
    c13,c14,c15,c16 = st.columns(4)
    c17,c18,c19,c191 = st.columns(4)
    c192,c193,c194,c195 = st.columns(4)
    st.write('---')

    with c13:
        quimica_notas_11 = create_column_feature_grades_11('quimica_notas_11')
    with c14:
        disciplina_notas_11 = create_column_feature_grades_11('disciplina_notas_11')
    with c15:
        economia_notas_11 = create_column_feature_grades_11('economia_notas_11')
    with c16:
        espaniol_notas_11 = create_column_feature_grades_11('espaniol_notas_11')
    with c17:
        filosofia_notas_11 = create_column_feature_grades_11('filosofia_notas_11')
    with c18:
        ingles_notas_11 = create_column_feature_grades_11('ingles_notas_11')
    with c19:
        matematicas_notas_11 = create_column_feature_grades_11('matematicas_notas_11')
    with c191:
        fisica_notas_11 = create_column_feature_grades_11('fisica_notas_11')
    with c192:
        ciencias_sociales_notas_11 = create_column_feature_grades_11('ciencias_sociales_notas_11')


    # =======================
    # Grades 12
    # =======================
    st.subheader('Grades 12 📘')
    c20,c21,c22,c23,c24 = st.columns(5)
    c25,c26,c27,c28,c29 = st.columns(5)
    st.write('---')

    with c20:
        quimica_notas_12 = create_column_feature_grades_12('quimica_notas_12')
    with c21:
        disciplina_notas_12 = create_column_feature_grades_12('disciplina_notas_12')
    with c22:
        economia_notas_12 = create_column_feature_grades_12('economia_notas_12')
    with c23:
        espaniol_notas_12 = create_column_feature_grades_12('espaniol_notas_12')
    with c24:
        filosofia_notas_12 = create_column_feature_grades_12('filosofia_notas_12')
    with c25:
        ingles_notas_12 = create_column_feature_grades_12('ingles_notas_12')
    with c26:
        matematicas_notas_12 = create_column_feature_grades_12('matematicas_notas_12')
    with c27:
        fisica_notas_12 = create_column_feature_grades_12('fisica_notas_12')
    with c28:
        estadistica_notas_12 = create_column_feature_grades_12('estadistica_notas_12')
    with c29:
        ciencias_sociales_notas_12 = create_column_feature_grades_12('ciencias_sociales_notas_12')


    # =======================
    # absences
    # =======================
    st.subheader('Absences 🤷‍♂️')
    c30,c31= st.columns(2)
    st.write('---')

    with c30:
        ausencias_11 = create_column_feature_absences('ausencias_11')
    with c31:
        ausencias_12 = create_column_feature_absences('ausencias_12')



    # Create the dataframe with all inputs
    dict_input = {
    'biologia_sim': biologia_sim,'cts_sim': cts_sim,'competencias_ciudadanas_sim': competencias_ciudadanas_sim,'fisica_sim': fisica_sim,'ingles_sim': ingles_sim,
    'lectura_critica_sim': lectura_critica_sim,'matematicas_(espec)_sim': matematicas__espec__sim,'matematicas_(cuant)_sim': matematicas__cuant__sim,'quimica_sim': quimica_sim,
    'sociales_sim': sociales_sim,'math_psat': math_psat,'reading_and_writing_psat': reading_and_writing_psat,'quimica_notas_11': quimica_notas_11,
    'disciplina_notas_11': disciplina_notas_11,'economia_notas_11': economia_notas_11,'espaniol_notas_11': espaniol_notas_11,'filosofia_notas_11': filosofia_notas_11,
    'ingles_notas_11': ingles_notas_11,'matematicas_notas_11': matematicas_notas_11,'fisica_notas_11':fisica_notas_11,'ciencias_sociales_notas_11':ciencias_sociales_notas_11,'quimica_notas_12': quimica_notas_12,'disciplina_notas_12': disciplina_notas_12,
    'economia_notas_12': economia_notas_12,'espaniol_notas_12': espaniol_notas_12,'filosofia_notas_12': filosofia_notas_12,'ingles_notas_12': ingles_notas_12,
    'matematicas_notas_12': matematicas_notas_12,'fisica_notas_12': fisica_notas_12,'estadistica_notas_12': estadistica_notas_12,'ciencias_sociales_notas_12':ciencias_sociales_notas_12,
    'ausencias_11': ausencias_11,'ausencias_12': ausencias_12, 
    }

    df_input = pd.DataFrame([dict_input])
    st.subheader('Table of your inputs: ')
    st.dataframe(df_input)
    st.write('---')
    colp1,colp2,colp3,colp4 = st.columns(4)
    
    # ===================
    # Button Prediction
    # ===================
    with colp1:
        prediction = st.button('Make Prediction')
                
    if prediction:
        st.balloons()
        prediccion = make_prediction(df_input,model)
            








