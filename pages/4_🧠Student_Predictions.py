import streamlit as st


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
# hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

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

st.title('Student Predictions 🧠')
st.write('---')

col1,col2,col3 = st.columns(3)
st.write('---')
col4,col5 = st.columns(2)
col6,col7 = st.columns(2)
with col2:
    student_code = st.text_input('Enter Student Code', placeholder = '20210122')

with col4:
    put_graph(file_path = './graphs/spider_graph.png', title = 'Spyder Chart')
with col5:
    put_graph(file_path = './graphs/kmeans_graph.jpeg', title = 'Segmentation Chart')
with col6:
    put_graph(file_path = './graphs/line_graph.png', title = 'Line Chart')