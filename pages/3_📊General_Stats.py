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

st.title('General Statistics 📊')
st.write('---')


col1,col2 = st.columns(2)
st.write('---')
col3,col4 = st.columns(2)

with col1:
    put_graph(file_path = './graphs/bar_graph.png', title = 'Bar Chart')
with col2:
    put_graph(file_path = './graphs/horizontal_bar_graph.png', title = 'Horizontal Chart')


with col3:
    put_graph(file_path = './graphs/heatmap_graph.png', title = 'Correlation HeatMap')
with col4:
    put_graph(file_path = './graphs/correlation_graph.png', title = 'Correlation Scatter Chart')

