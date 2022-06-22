import pandas as pd
import sqlalchemy
import streamlit as st
import plotly.express as px
from PIL import Image
from sqlalchemy import create_engine
import plotly.graph_objects as go
import requests
import json
import base64
from tpv_mega import *


# ============
# FUCTIONS
# ============

def telegram_send_message(text_to_send:str):
    TOKEN = '1914385797:AAEM7u0UCv8a2ctjNFm8NaGkDhFm7Fd7FKI'
    CHAT_ID = '-575502462'
    url_send_message = f'https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={text_to_send}'
    r = requests.get(url_send_message)
    if r.status_code == 200:
        print('Good \U0001F600')
    else:
        print('Bad \U0001F612')


@st.cache(show_spinner=False)
def ccms_autentication(user:str ,password:str):

    url = 'https://oauth.teleperformance.co/api/oauthlogin'

    cadena = '{"user": ' + f'"{user}","pass":' + '"' + password + '"}'

    enc = cadena.encode()
    x = base64.b64encode(enc).decode('utf-8')
    cadena_codificada = 't' + x

    data = {
        "body": cadena_codificada,
        "project": "Test",
        "ip": "123d4",
        "uri": "aptp",
        "size": 0
        }

    try:
        r = requests.post(url, data=data)
        json_gen = json.loads(r.text)
    except:
        pass

    if r.status_code == 200:
        name = str(json_gen['data']['nombre'])
        return  True,name
    else:
        name = 'Invalid credencials'
        return False,name

def connect_with_sales_aws_db()->sqlalchemy:

    host = 'sales-database.c9zdhywb49dn.us-east-2.rds.amazonaws.com'
    user = 'javier_admin'
    password = 'Mithbuster1*'
    port = 3306
    # Database connection URI structure
    # [DB_TYPE]+[DB_CONNECTOR]://[USERNAME]:[PASSWORD]@[HOST]:[PORT]/[DB_NAME]
    conn = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/interactions_DB') # connect to server

    return conn

@st.cache(show_spinner=False)
def load_dataframe(table_name:str)->pd.DataFrame:

    conn = connect_with_sales_aws_db()
    QUERY = f'SELECT * FROM {table_name}'
    df = pd.read_sql(QUERY,conn)
    
    return df
    
def text_center(texto:str):

    text_html = f'<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)

def text_center_light(texto:str):

    text_html = f'<div style="text-align: center;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)
@st.cache(show_spinner=False)
def extract_forms_information()->pd.DataFrame:
    
    end_point = 'https://ws.tpdigitgen.tech/'    
    data = {'lob': "All",
    'task': "all",
    'type': "Production",
    '_task': "leads2",'_post':True}
    
    r = requests.post(end_point,data)
    json_gen = json.loads(r.text)
    
    lista_diccionarios = []
    for i in json_gen['data']:
        diccionario = {}
        edp = i['edp']
        data = i['data'][0]['data']
        diccionario.update({'edp':edp})
        diccionario.update(data)
        lista_diccionarios.append(diccionario)
        
    df = pd.DataFrame(lista_diccionarios)
    df['gmt'] = pd.to_datetime(df['gmt'])
    df['date'] = df['gmt'].apply(lambda x: x.strftime('%m-%d-%Y'))
    df['date'] = pd.to_datetime(df['date'])
    df['date']  = df['date'].dt.date
    df.fillna('')
    df['col42'] = df['col42'].apply(lambda x: str(x))
    
    return df

@st.cache(show_spinner=False,allow_output_mutation=True)
def download_tpv_mega():
    return tpv_mega()

@st.cache(show_spinner=False,allow_output_mutation=True)
def load_tpv_mega(table_name:str)->pd.DataFrame:

    conn = connect_with_sales_aws_db()
    QUERY = f'SELECT * FROM {table_name}'
    df = pd.read_sql(QUERY,conn)
    
    return df


    
text_html = '<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> Authentication </div>'
st.sidebar.markdown(text_html, unsafe_allow_html=True)
st.sidebar.write(' ')
login_expander = st.sidebar.expander('Login')
with login_expander:
    user = login_expander.text_input('User NET')
    password = login_expander.text_input('Password',type="password")
    login_ckeck_box = login_expander.checkbox('Login')

if login_ckeck_box == True: # Si el logeo fue exitoso
    if user == 'ana' and password == 'ana':
        login = True
        name = 'Ana Maria Rodriguez'
    else:
        login,name = ccms_autentication(user,password)
        login_expander.success(f'User: {name}')
    if name != 'Invalid credencials' :
        pass
        # telegram_send_message(f'El usuario {name} hizo login')
    
    if login == True:

        bootstrap = st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)
        icons = st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css">', unsafe_allow_html=True)
        st.title('Power Choice Now')
        st.write('---')
        hide_streamlit_style = """<style>footer{visibility: hidden;}</style>"""
        st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

        # Start the magic
        with st.spinner('Loading Livevox interactions...'):
            pcn_df_original = load_dataframe('pcn_calls')
        pcn_df = pcn_df_original.copy()
        
        # with st.spinner('Loading forms data...'):
        #     forms_df = extract_forms_information()

        services_counts = pcn_df['service'].value_counts(normalize=True).to_frame()#.reset_index()
        services_counts.reset_index(inplace=True)
        services_counts.columns = ['Service Name','Count']
        services_counts['Count'] = round(services_counts['Count'] * 100,1)
    
                    
        #------------- side bar 2
        # DATE PICKER
        st.sidebar.write('---')
        start_date_col,end_date_col = st.sidebar.columns(2)
        dates = pcn_df['date'].unique().tolist()
        with start_date_col:
            text_center('Start Date')
            date_start = start_date_col.date_input(' ',value=max(dates),min_value=min(dates),max_value=max(dates))
        
        with end_date_col:
            text_center('End Date')
            date_finish = end_date_col.date_input('  ',value=max(dates),min_value=min(dates),max_value=max(dates))
        st.sidebar.write('---')


        # st.sidebar.subheader('Livevox Interactions')
        text_html = '<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> Livevox Interactions </div>'
        st.sidebar.markdown(text_html, unsafe_allow_html=True)
        st.sidebar.write(' ')
        # Multiselect COLUMNS
        expander_columns_interactions = st.sidebar.expander('Select Columns LV Interactions')
        with expander_columns_interactions:
            columns_names = pcn_df.columns.tolist()
            container_column = expander_columns_interactions.container()
            all_columns = expander_columns_interactions.checkbox("Select all columns")
            if all_columns:
                columns_names_multiselect = container_column.multiselect('Columns:',columns_names, default = columns_names)
            else:
                
                columns_names_multiselect = container_column.multiselect('Columns:',columns_names, default = ['ID','phone_number','service','livevox_result','date','agent_id','filename','account_number'])

        #------------- side bar 3
        # Multiselect DISPOSITION CODE
        expander_dispo_code = st.sidebar.expander('Filter by: Disposition Code')
        with expander_dispo_code:
            disposition_codes = pcn_df['livevox_result'].unique().tolist()
            container_dispo = expander_dispo_code.container()
            all = expander_dispo_code.checkbox("Select all disposition codes")
            if all:
                dispo_cod_multiselect = container_dispo.multiselect('Disposition Codes:',disposition_codes, default = disposition_codes)
            else:
                dispo_cod_multiselect = container_dispo.multiselect('Disposition Codes:',disposition_codes, default = ['AGENT - CUST RPC 7','AGENT - CUST WPC 7'] )

        #------------- side bar 4
        # Multiselect SERVICE
        expander_service_name = st.sidebar.expander('Filter by: Service Name')
        with expander_service_name:
            services = pcn_df['service'].unique().tolist()
            container_service = expander_service_name.container()
            services_multiselect = container_service.multiselect('Service Name:',services, default = services)
        

        # FILTERING ACORDING TO LEFT FILTERS

        # FILTER DF BASE ON SELECTION
        mask = ((pcn_df['date'].between(date_start,date_finish)) & (pcn_df['livevox_result'].isin(dispo_cod_multiselect)) & (pcn_df['service'].isin(services_multiselect)))
        pcn_df = pcn_df[mask]
        pcn_df = pcn_df[columns_names_multiselect]
        pcn_df = pcn_df.drop_duplicates(subset = ['phone_number','date'], keep= 'last')
        total_rows = len(pcn_df)

        
        sales = pcn_df_original.copy()
        mask_sales = ((sales['date'].between(date_start,date_finish)) & (sales['service'].isin(services_multiselect)))
        sales = sales[mask_sales]
        sales = sales[sales['livevox_result'].isin(['AGENT - CUST WPC 7','AGENT - CUST RPC 7'])]
        sales = sales.drop_duplicates(subset = ['phone_number','date'], keep= 'last')
        

        # mask_forms = ((forms_df['date'].between(date_start,date_finish)) & (forms_df['lob'] == 'PCN'))
        # forms_df = forms_df[mask_forms]

        agents_sales_counts = sales['agent_id'].value_counts().to_frame().reset_index().head(5)
        agents_sales_counts.columns = ['agent_id','count']
        agents_sales_counts = agents_sales_counts.sort_values(by = ['count'], ascending=False)


        # RESUMEN CONTAINER
        review_container = st.container()
        tab = '&nbsp;&nbsp;&nbsp;&nbsp;'
        with review_container:
            # best_agent = agents_sales_counts[agents_sales_counts['count'] == agents_sales_counts['count'].max()]['agent_id'][0].lower()
            
            
            col_sales,col_verified,col_declined,col_expired = st.columns((1,1,1,1))
            with col_sales:
                st.markdown(f'<p class="h4" style = "text-align: center;">LV Sales: <b style = "color: #C84B31;">{len(sales)}</b></p>', unsafe_allow_html=True)

            # with col_forms:
            #     st.markdown(f'<p class="h4" style = "text-align: center;">Forms: <b style = "color: #C84B31;">{len(forms_df)}</b></p>', unsafe_allow_html=True)

            # with col_agent:
            #     st.markdown(f'<p class="h4" >Best Agent: <b style = "color: #C84B31;text-align: center;">{best_agent}</b></p>', unsafe_allow_html=True)

            with col_verified:
                verified_text_field = st.empty()

            with col_declined:
                declined_text_field = st.empty()

            with col_expired:
                expired_text_field = st.empty()

            verified_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Verified: <b style = "color: #C84B31;">...</b></p>', unsafe_allow_html=True)
            declined_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Declined: <b style = "color: #C84B31;">...</b></p>', unsafe_allow_html=True)
            expired_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Expired: <b style = "color: #C84B31;">...</b></p>', unsafe_allow_html=True)
            st.write('---')

        # Interaction Tables
        with st.expander("Livevox interactions", expanded = True):
            # Put DATAFRAME in page
            
            st.dataframe(pcn_df,height=500)
            st.caption('*All the duplicates rows was deleted by: phone number and date.')
            st.markdown(f'Total results: {total_rows}')
            st.write('---')

        

        # Sales by date expander
        with st.expander("Charts"):

            line_chart_col,bar_chart_col,pie_chart_col = st.columns((2,1,1))
            # line_chart_col = st.container()

            with line_chart_col:
                # st.write('Sales by date')
                # st.subheader('Sales by Date')
                text_center('Sales by Date')
                sales = pcn_df[pcn_df['livevox_result'].isin(['AGENT - CUST WPC 7','AGENT - CUST RPC 7'])]
                sales_by_date = sales['date'].value_counts(dropna=False).to_frame().reset_index()
                sales_by_date.columns = ['date','# sales']
                sales_by_date['date'] = pd.to_datetime(sales_by_date['date'])
                sales_by_date.sort_values(by=['date'], ascending=True,inplace=True)
                fig = px.line(sales_by_date, x="date", y="# sales")#,width=1240, height=500)
                st.plotly_chart(fig,use_container_width = True)

            
            # Bar Chart
            with bar_chart_col:
                # st.subheader('Top 5 Agents')
                agents_sales_counts = pcn_df['agent_id'].value_counts().to_frame().reset_index().head(5)
                agents_sales_counts.columns = ['agent_id','count']
                agents_sales_counts = agents_sales_counts.sort_values(by = ['count'], ascending=False)
                text_center('Top 5 Agents')
                colors = ['lightslategray',] * 5
                colors[0] = 'crimson'

                bar_chart_plot = go.Figure(data=[go.Bar(
                    x=agents_sales_counts['agent_id'],
                    y=agents_sales_counts['count'],
                    marker_color=colors)],)
                # bar_chart_plot.update_layout(width = 700, height=500)
                bar_chart_plot.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    })
                st.plotly_chart(bar_chart_plot,use_container_width = True)
                st.caption('*All the graphs will vary according to the left filters')

            # Donut Chart
            with pie_chart_col:

                # st.subheader('Service proportion')
                text_center('Service Proportion')
                
                # PIE CHART
                services_counts = pcn_df['service'].value_counts().to_frame()#.reset_index()
                services_counts.reset_index(inplace=True)
                services_counts.columns = ['Service Name','Count']
                pie_chart_for_sales = go.Figure(data=[go.Pie(labels=services_counts['Service Name'],values=services_counts['Count'], hole=.6)])
                # pie_chart_for_sales = px.pie(dispo_counts, title = 'Service name distribution', values = 'Count',names = 'Service Name',width=600,height=400)
                # pie_chart_for_sales.update_layout(height=500)
                pie_chart_for_sales.update_layout(legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.1,
                                            xanchor="right",
                                            x=0.9
                                        ))

                pie_chart_for_sales.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    })
                st.plotly_chart(pie_chart_for_sales,use_container_width = True)
    
        
        #------------- side bar 5
        # WEB FORMS
        # st.sidebar.write('---')
        # text_html = '<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> Web Forms </div>'
        # st.sidebar.markdown(text_html, unsafe_allow_html=True)
        # st.sidebar.write(' ')
        # # sidebar expander Forms
        # expander_forms_columns = st.sidebar.expander("Select Web Form Columns", expanded = False)
        
        # with expander_forms_columns:

        #     # Multiselect COLUMNS
        #     forms_columns_names = forms_df.columns.tolist()
        #     forms_container_column = expander_forms_columns.container()
        #     forms_all_columns = expander_forms_columns.checkbox("Select all columns ")
        #     if forms_all_columns:
        #         forms_columns_names_multiselect = forms_container_column.multiselect('Columns:',forms_columns_names, default = forms_columns_names)
        #     else:
        #         forms_columns_names_multiselect = forms_container_column.multiselect('Columns:',forms_columns_names, default = ['edp', 'ip','lob', 'domain', 'gmt','city', 'state', 'score', 'source', 'zip', 'first', 'last', 'email','phoneNumber', 'ipClass','date'])
        
        
        
        # # form expander table
        # with st.expander("Web Form"):
        #     forms_df = forms_df[forms_columns_names_multiselect]
        #     mask = ((forms_df['date'].between(date_start,date_finish)) & (forms_df['lob'] == 'PCN'))
        #     forms_df = forms_df[mask]
        #     forms_total_rows = len(forms_df)
        #     st.dataframe(forms_df)
        #     st.markdown(f'Total results: {forms_total_rows}')


        logtxtbox = st.empty()
        logtxt = 'start'
        # logtxtbox.text_area("Logging: ",logtxt, height = 1)
        
        with st.expander('TPV'):

            with st.spinner('Loading TPV Information...'):
                df_tpv_mega = load_tpv_mega('tpv_mega')
                df_tpv_mega['SoldDateTime'] = pd.to_datetime(df_tpv_mega['SoldDateTime'])
                df_tpv_mega['SoldDateTime'] = df_tpv_mega['SoldDateTime'].dt.date
                mask_tpv = df_tpv_mega['SoldDateTime'].between(date_start,date_finish)
                df_tpv_mega = df_tpv_mega[mask_tpv]
                
                st.dataframe(df_tpv_mega)
                verified = len(df_tpv_mega[df_tpv_mega['Status'] == 'Verified'])
                declined = len(df_tpv_mega[df_tpv_mega['Status'] == 'Declined'])
                expired = len(df_tpv_mega[df_tpv_mega['Status'] == 'Expired'])

                # tpv_text_field.markdown(f'<p class="h4" >: <b style = "color: #C84B31;font-size :1.5rem;">{best_agent}</b></p>', unsafe_allow_html=True)
                declined_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Declined: <b style = "color: #C84B31;">{declined}</b></p>', unsafe_allow_html=True)
                verified_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Verified: <b style = "color: #C84B31;">{verified}</b></p>', unsafe_allow_html=True)
                expired_text_field.markdown(f'<p class="h4" style = "text-align: center;">TPV-Expired: <b style = "color: #C84B31;">{expired}</b></p>', unsafe_allow_html=True)
                
                st.markdown(f'Total Results: <b>{len(df_tpv_mega)}</b>', unsafe_allow_html=True)

        #------------- side bar 6
        # TPV
        st.sidebar.write('---')
        text_html = '<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> Third Party Verification </div>'
        with st.sidebar.markdown(text_html, unsafe_allow_html=True):
            st.sidebar.write(' ')
            col_1,col_2,col_3 = st.sidebar.columns((0.5,1.2,0.5))

            with col_2:
                button_tpv_update = col_2.button('Get TPV updated')
                    

        if button_tpv_update:
            with st.spinner('Loading TPV...'):
                tpv_today = tpv_mega_today()
            tpv_today['SoldDateTime'] = pd.to_datetime(tpv_today['SoldDateTime'])
            
            verified_today = len(tpv_today[tpv_today['Status'] == 'Verified'])
            declined_today = len(tpv_today[tpv_today['Status'] == 'Declined'])
            expired_today = len(tpv_today[tpv_today['Status'] == 'Expired'])
            pending_today = len(tpv_today[tpv_today['Status'] == 'Pending'])

            tpv_today_container = st.container()
            with tpv_today_container:
                st.markdown(f'Update: {datetime.now().strftime("%A, %x %X")}')
                st.write('  ')
                v_today,p_today,d_today,e_today = st.columns(4)
                
                with d_today:
                    st.markdown(f'<p class="h4" style = "text-align: center;">TPV-Declined: <b style = "color: #233E8B;">{declined_today}</b></p>', unsafe_allow_html=True)
                with v_today:
                    st.markdown(f'<p class="h4" style = "text-align: center;">TPV-Verified: <b style = "color: #233E8B;">{verified_today}</b></p>', unsafe_allow_html=True)
                with e_today:
                    st.markdown(f'<p class="h4" style = "text-align: center;">TPV-Expired: <b style = "color: #233E8B;">{expired_today}</b></p>', unsafe_allow_html=True)
                with p_today:
                    st.markdown(f'<p class="h4" style = "text-align: center;">TPV-Pending: <b style = "color: #233E8B;">{pending_today}</b></p>', unsafe_allow_html=True)
            
            
        

else: # si el logeo no fue exitoso.
    st.title(':worried: Ops!')
    st.warning('Check your credentials, please!')
    st.write('---')
