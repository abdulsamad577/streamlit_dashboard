from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
# import pandas_profiling

st.markdown('''
# **Exploratory Data Analysis Web Application**
            This app is developed by Data Scientist, Samad called **EDA App**
            ''')
with st.sidebar.header("Upload your dataset(.CSV)"):
    upload_file=st.sidebar.file_uploader("Upload Your File",type=['csv'])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown("[Example csv file](df)")
if upload_file is not None:
 
    def load_csv():

        csv=pd.read_csv(upload_file)
      
        return csv
    df=load_csv()
    st.write(df.head())
    pr=ProfileReport(df,explorative=True)
    st.header("**Input DF**")
    st.write(df)
    st.write("---")
    st.header("**Profiling reform with Pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting of CSV file...")

    