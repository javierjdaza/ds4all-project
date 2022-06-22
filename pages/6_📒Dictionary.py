import streamlit as st


st.set_page_config(page_title='DS4A: Marymount',layout='wide',page_icon="./images/marymount_favicon.ico")#,initial_sidebar_state="collapsed")
# hide_streamlit_style = """<style>#MainMenu {visibility:hidden;}footer {visibility:hidden;}</style>"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.image('./images/marymount_logo.png')
with col4:
    st.image('./images/correlation.png')
st.write('----')

st.title('Data Dictionary 📒')
st.write('----')

st.subheader('GRADES 📓')
st.markdown('''
**DEFINITIVE GRADES:** a CSV file that contains definitive notes by each subject for every student of each grade from 2017-2018 until 2021-2022 school year .
1. **CODIGO:** The student ID. Throughout their school years, every student has the same code.  \n   
2. **AÑO ESCOLAR:** The school year in which the subject is evaluated.  \n   
3. **GRADO:** The grade to which student belongs in each school year.  \n  
4. **ASIGNATURA:** The subject evaluated.  \n  
5. **NUMERO DE PERIODO:** Academic quarter in which the subject is evaluated.  \n  
6. **RESULTADO:** The result obtained by each student in each quarter for every subject.''')
st.write('---')

st.subheader('PSAT TEST 📋')
st.markdown('''
**PSAT TEST:** the Preliminary SAT/National Merit Scholarship Qualifying Test (NMSQT=is a standardized test targeting 10th and 11th graders in the USA, in this case for the Marymount Highschool the test targeting only 10th. This test assesses math and reading and writing skills, frequently people with the highest scores receive scholarship offers from a variety of organizations. The data resides in a CSV file which contains the math, reading and writing (in the same category) and global results for every student of 10th grade from 2017-2018 until 2021-2022 school year.

1. **CODIGO:** The student ID.  \n  
2. **AÑO ESCOLAR:** The school year in which the test is taken.  \n  
3. **GRADO:** The student's grade in which the test was taken.  \n  
4. **ASIGNATURA:** The subject evaluated.  \n  
5. **RESULTADO:** The result obtained by each student in each subject, this also contains the global result by each student.''')
st.write('---')

st.subheader('DRILL TEST OF PRUEBAS SABER 11 📝' )
st.markdown('''
**DRILL TEST OF PRUEBAS SABER 11:** The School prepares 11th and 12th grade students for the official state test "PRUEBAS SABER 11" through a drill test targeted at least twice a year. This dataset contains the results by each drill test per student, key subjects, grade and school year.

1. **CODIGO:** The student ID.  \n  
2. **AÑO ESCOLAR:** The school year in which the drill test is taken.  \n  
3. **GRADO:** The student's grade in which the test was taken.  \n  
4. **ASIGNATURA:** The subject evaluated.  \n  
5. **NUMERO DE PRUEBA:** number of the drill test taken.  \n  
6. **RESULTADO:** The result obtained by number of drill test, subject, student, grade and school year.''')
st.write('---')


st.subheader('SABER 11 Test 📔' )
st.markdown('''
**RESULTS OF PRUEBAS SABER 11 TEST:** Every year in Colombia, all 11th grade students (12th grade at Marymount school) are required to take the "Pruebas Saber 11" test, which is used to assess the academic quality of schools and as a criterion for acceptance into universities and other institutions. This csv file contains the results of "Pruebas Saber 11" test by student, subject and school year.

1. **CODIGO:** The student ID.  \n  
2. **AÑO ESCOLAR:** The school year in test was taken.  \n  
3. **GRADO:** The student's grade in which the test is taken.  \n  
4. **ASIGNATURA:** The subject evaluated.  \n  
5. **RESULTADO:** The result obtained by subject, student and school year. This also contains the global result by each student.''')
st.write('---')


st.subheader('Students 🙋‍♂️🙋‍♀️' )
st.markdown('''
**STUDENTS LIST:** a CSV file that contains all students who have been in the high school from 2017-2018 until 2021-2022 school year.

1. **CODIGO:** The student ID.  \n  
2. **AÑO ESCOLAR:** the first school year for which the student's test records or grades are available.  \n  
3. **GRADO:** The first grade for which the student's test records or grades are available.''')
st.write('---')
