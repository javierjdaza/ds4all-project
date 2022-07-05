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

def put_graph(file_path:str, title:str):
    st.subheader(title)
    st.markdown('#')
    st.image(file_path)

st.title('General Statistics 📊')
st.write('---')


col1,col2 = st.columns(2)
st.write('---')
col3,col4 = st.columns(2)
st.write('---')
col5,col6 = st.columns(2)
st.write('---')
col7,col8 = st.columns((20,1))
st.write('---')
col9,col10 = st.columns(2)
st.write('---')
col11,col12 = st.columns((20,1))

with col1:
    put_graph(file_path = './charts/Results of "Saber 11" test per school year.png', title = 'Results of "Saber 11" test per school year')
with col2:
    put_graph(file_path = './charts/Global results of "Pruebas saber 11" test per school year.png', title = 'Global results of "Pruebas saber 11" test per school year')
    


with col3:
    put_graph(file_path = './charts/Comparison Global results of "Saber 11" test.png', title = 'Comparison Global results of "Saber 11" test')
with col4:
    put_graph(file_path = './charts/Number of students per School year in "Saber 11" test.png', title = 'Number of students per School year in "Saber 11" test')

with col5:
    put_graph(file_path = './charts/Results of drill test per Subjects by school year.png', title = 'Results of drill test per Subjects by school year')
with col6:
    put_graph(file_path = './charts/Comparative of Grade 11th vs Grade 12th Drill test results.png', title = 'Comparative of Grade 11th vs Grade 12th Drill test results')

with col7:
    put_graph(file_path = './charts/grades_reports.png', title = 'Grades Reports')

with col9:
    put_graph(file_path = './charts/PSAT results per School year.png', title = 'PSAT results per School year')
with col10:
    put_graph(file_path = './charts/Combined results of PSAT test.png', title = 'Combined results of PSAT test')

with col11:
    put_graph(file_path = './charts/Correlations between "Saber 11" test results and performance of subjects, drill test and PSAT test.png', title = 'Correlations between "Saber 11" test results and performance of subjects, drill test and PSAT test')

