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

st.title('The Team 🎖')
st.write('---')

def text_center(texto:str):

    text_html = f'<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)
def put_image(file_path:str, caption:str):
    st.image(file_path, caption = caption)

img1,img2,img3,img4 = st.columns(4)
st.markdown('#')
img5,img6,img7,img8 = st.columns(4)

with img1:
    put_image(file_path = './images/team/Andres Sanchez.png', caption = 'Andres Sanchez')
with img2:
    put_image(file_path = './images/team/Ivan Santiago Bello Quete.png', caption = 'Ivan Santiago Bello Quete')
with img3:
    put_image(file_path = './images/team/Geraldine Berrio Sanchez.png', caption = 'Geraldine Berrio Sanchez')
with img4:
    put_image(file_path = './images/team/Javier Daza Olivella.png', caption = 'Javier Daza Olivella')
with img5:
    put_image(file_path = './images/team/Sebastian Bedoya.png', caption = 'Sebastian Bedoya')
with img6:
    put_image(file_path = './images/team/Juan Andres Gonzalez Urquijo.png', caption = 'Juan Andres Gonzalez Urquijo')
with img7:
    put_image(file_path = './images/team/Jorge Andres Alzate Hoyos - TA.png', caption = 'Jorge Andres Alzate Hoyos - TA')
with img8:
    put_image(file_path = './images/team/Gonzalo Cossio - TA.png', caption = 'Gonzalo Cossio - TA')
