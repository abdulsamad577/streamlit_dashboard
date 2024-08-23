import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#make containers
header= st.container()
data=st.container()
features=st.container()
model_training=st.container()



with header:
    st.title('Titanic App')
    st.text('In the project we will work on Titanic data')


with data:
    st.header("The Ship has Sunk")
    st.text('we will discuss about the Titanic accident dataset.')
    
    #Import data
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head())
    st.subheader("How many people in the Ship?")
    st.bar_chart(df['sex'].value_counts())
    
    #other plot
    st.subheader("Class k hssab se faraq")
    st.bar_chart(df['class'].value_counts())

    #barplot
    st.bar_chart(df['age'].sample(10)) # or .head(10)


with features:
    st.header("These are our app features")
    st.text("We will add the features of the application in this container")
    st.markdown("1.**Feature 1:** Thsi will tell us something")
    st.markdown("2.**Feature 2:** Thsi will tell us something")


with model_training:
    st.header("What happened to the people on the ship? -Model Training....")
    st.text("In this, we will increase or decrease our parameter...")

    #making columns
    input,display=st.columns(2)

    #Pehly column mei ap k selection points hun
    max_depth=input.slider("How many people do you know?",min_value=10,max_value=100, value=20,step=5)

    #n_estimators
    n_estimators=input.selectbox("how many tree should be there in a RF?",options=[50,100,200,300,'No Limit'])

    #adding list of features
    input.write(df.columns)
    #input features from user
    input_features=input.text_input('Which feature we should use?')
    if input_features in df.columns:


        # Machine Learning Model
        model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
        # We used a condition
        if n_estimators=='No Limit':
            model=RandomForestRegressor(max_depth=max_depth)
        else:
            model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
        # Define X and Y
        X=df[[input_features]]
        y=df[['fare']]

    
        #fit our model
        model.fit(X,y)
        pred=model.predict(X)

        #Display metrices
        display.subheader("Mean absolute error of the model is: ")
        display.write(mean_absolute_error(y,pred))
        display.subheader("Mean Squared error of the model is: ")
        display.write(mean_squared_error(y,pred))
        display.subheader("R Squared score of the model is: ")
        display.write(r2_score(y,pred))
    else:
        input.write("Selected a column.....")

