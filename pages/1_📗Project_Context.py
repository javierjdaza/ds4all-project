import streamlit as st


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png')
with col4:
    st.image('./images/correlation.png')
st.write('----')


st.title('Project Context 📗')
st.write('----')
st.markdown('''
**Business Context:**  \n  Identify those predictive factors of success and risk (“associated factors”) 
for the results of the Saber 11 tests of our students who are from sixth to eleventh 
grade of high school; where a predictive model can be developed in which the associated 
factors that may incur in the results of the Saber Tests can be identified in each cohort, 
taking into account all their academic grades, as well as the grades of the standardized tests they have taken.''')
st.markdown('#')

st.markdown('''
**Business Problem:**  \n  Define an Alert System which enables the Marymount school to identify areas for improvement in order to avoid detecting poor results on state test (Pruebas saber 11) for 10th to 12th grade students.''')
st.markdown('#')
st.markdown('''**Analytical Context.**   \n  The school has 5 csv files containing details about results of "Prueba Saber 11" test,
preparations for "Pruebas Saber 11", PSAT test, grades for all subject and student's list from 2017-2018 until 2021-2022 school year.''')