import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.write('''
         # Random Forest Classifier App
         ##### Made by Samad
         This app predict the type of iris based on Sepal Length, Sepal Width, Petal Length, Petal Width
         ''')

st.sidebar.header("Change IRIS Parameters:")
sepal_length=st.sidebar.slider('Sepal Length',0.0,7.9,1.0)
sepal_width=st.sidebar.slider("Sepal Width",0.0,4.4,1.0)
petal_length=st.sidebar.slider("Petal Length",0.0,7.0,2.0)
petal_width=st.sidebar.slider("Petal Width",0.0,2.5,1.1)
data={
    'sepal_length': sepal_length,
    'sepal_width':sepal_width,
    'petal_length':petal_length,
    'petal_width':petal_width
}
df=pd.DataFrame(data,index=[0])

st.subheader("IRIS Parameters:")
st.write(df)

iris=sns.load_dataset("iris")
# st.write(iris.describe())

X=iris[['sepal_length','sepal_width','petal_length','petal_width']]
y=iris['species']

model=RandomForestClassifier()
model.fit(X,y)

prediction=model.predict(df)
prediction_proba=model.predict_proba(df)

st.subheader("Class label and their coresponding index numnber")
value=iris['species'].unique()
st.write(pd.DataFrame(value))

st.subheader('Prediction:')

st.write(prediction[0])

st.subheader("Prediction Probability:")
st.write(prediction_proba)

st.subheader("IRIS Plot:")
fig=px.scatter(iris,x='sepal_length',y='petal_length',color='species')
st.plotly_chart(fig)

