import streamlit as st
import pandas as pd


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 





col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png')
with col4:
    st.image('./images/correlation.png')
st.write('----')


st.title('Data 📁')
st.write('---')
st.markdown('#')
@st.cache(show_spinner=False)
def load_dataframe(file_path:str)->pd.DataFrame:

    df = pd.read_csv(file_path, encoding = 'latin1', sep = ';')
    
    return df

def display_dataframe(df:pd.DataFrame,name_table:str):
    columns = [ i.lower().replace('ï»¿','').replace('aã\x91o','año').replace('a\x84o','año') for i in df.columns]
    df.columns = columns
    st.subheader(name_table)

    st.dataframe(df)
    
    st.caption(f'columns: {df.shape[0]}  \n  rows: {df.shape[1]}')
    st.write('---')


df1,df2 = st.columns(2)
df3,df4 = st.columns(2)
df5,df6 = st.columns(2)


with df1:
    estudiantes = load_dataframe('./data/Listado_estudiantes.csv')
    display_dataframe(df = estudiantes,name_table = 'Students 🙋‍♂️🙋‍♀️')
    
with df2:
    pruebas_psat = load_dataframe('./data/Pruebas_PSAT.csv')
    display_dataframe(df = pruebas_psat,name_table = 'Psat Tests 📋')

with df3:
    pruebas_saber = load_dataframe('./data/Pruebas_Saber_11.csv')
    display_dataframe(df = pruebas_saber,name_table = 'Saber 11 Tests 📔')
    
with df4:
    simulacros = load_dataframe('./data/Simulacro_pruebas_saber_11.csv')
    display_dataframe(df = simulacros,name_table = 'Drill Tests 📝')
    
with df5:
    definitivas = load_dataframe('./data/Definitivas_asignaturas.csv')
    display_dataframe(df = definitivas,name_table = 'Grades 📓')
    
with df6:
    ausencias = load_dataframe('./data/Ausencias.csv')
    display_dataframe(df = definitivas,name_table = 'Absences 🤷‍♂️')
    
    