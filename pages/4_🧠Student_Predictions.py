import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu
from students_10 import student_10
from students_11 import student_11
from students_12 import student_12
from students_pred import student


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


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


col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png')
with col4:
    st.image('./images/correlation.png')
st.write('----')

st.title('Student Predictions 🧠')
st.write('---')

selected = option_menu(menu_title=None,options = ["Student Prediction","Students Grade: 10", 'Students Grade: 11','Students Grade: 12'], 
icons = ["columns-gap","file-earmark-person", "file-earmark-person","file-earmark-person",] ,default_index = 0,orientation="horizontal")
st.write('---')


if selected == 'Students Grade: 10':
    student_10()
if selected == 'Students Grade: 11':
    student_11()
if selected == 'Students Grade: 12':
    student_12()
if selected == 'Student Prediction':
    student()

            








