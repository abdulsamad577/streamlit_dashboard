import seaborn as sns
import streamlit as st
st.header("This is my First Streamlit Application!")
df=sns.load_dataset('titanic')
st.write(df.head())
st.bar_chart(df[['age','survived']])
st.line_chart(df['age'])