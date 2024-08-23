import streamlit as st
import plotly.express as px
import pandas as pd

#import dataset
st.title("plotly and streamlit")
df=px.data.gapminder()
st.write(df)
st.write(df.columns)

st.write(df.describe())
year_option=(df.year.unique().tolist())
year=st.selectbox("Which year should you plot? ",year_option)
# df=df[df['year']==year]
# st.write(df)

fig=px.scatter(df,x='gdpPercap',y='lifeExp',size='pop',color='continent',hover_name='continent',log_x=True,size_max=60,
               range_x=[100,100000],range_y=[20,90],
               animation_frame='year',animation_group='country')


st.write(fig)