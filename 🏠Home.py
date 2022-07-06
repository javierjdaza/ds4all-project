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


@st.cache(show_spinner=False)
def load_dataframe(file_path:str)->pd.DataFrame:

    df = pd.read_csv(file_path)
    
    return df
    
def text_center(texto:str):

    text_html = f'<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)

def text_center_light(texto:str):

    text_html = f'<div style="text-align: center;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)



st.title('About the School')
st.write('----')
st.write("""The Marymount School -Barranquilla is a private, mixed and bilingual institute in Barranquilla City.\n 
The objective of the school is to educate leaders who are capable of embarking on a transformative life project in a 
globalized society through high quality education. One of the most important meassures of success meters is the results 
of 'Pruebas Saber 11', a standardized state test administered by the Colombian government to all students in grade 11 
(grade 12 at Marymount School), which evaluates educational quality and is also used as a criterion for acceptance into universities
 and other higher education institutions. \n 
 In the last three years the students performance has been decreasing in this test and the 
 school wants to improve and make decisions to help its students with their learning outcomes.""")

# st.sidebar.header("Mapping Demo")
    
