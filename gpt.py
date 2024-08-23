import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make containers
header = st.container()
data = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Titanic App')
    st.text('In this project, we will work on Titanic data')

with data:
    st.header("The Ship has Sunk")
    st.text('We will discuss the Titanic accident dataset.')
    
    # Import data
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head())
    st.subheader("How many people on the Ship?")
    st.bar_chart(df['sex'].value_counts())

    # Other plot
    st.subheader("Class wise distribution")
    st.bar_chart(df['class'].value_counts())

    # Barplot
    st.bar_chart(df['age'].sample(10))  # or .head(10)

with features:
    st.header("These are our app features")
    st.text("We will add the features of the application in this container")
    st.markdown("1. **Feature 1:** This will tell us something")
    st.markdown("2. **Feature 2:** This will tell us something")

with model_training:
    st.header("What happened to the people on the ship? - Model Training...")
    st.text("In this, we will increase or decrease our parameters...")

    # Making columns
    input, display = st.columns(2)

    # Column for parameter selection
    max_depth = input.slider("Max depth of the trees?", min_value=10, max_value=100, value=20, step=5)
    n_estimators = input.selectbox("How many trees should be there in the Random Forest?", options=[50, 100, 200, 300, 'No Limit'])

    # Adding list of features
    input.write(df.columns)
    
    # Input features from user
    input_features = input.text_input('Which feature should we use?')

    if input_features in df.columns:
        # Machine Learning Model
        if n_estimators == 'No Limit':
            model = RandomForestRegressor(max_depth=max_depth)
        else:
            model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

        # Define X and Y
        X = df[[input_features]]
        y = df['fare']

        # Fit our model
        model.fit(X, y)
        pred = model.predict(X)

        # Display metrics
        display.subheader("Mean Absolute Error of the model:")
        display.write(mean_absolute_error(y, pred))
        display.subheader("Mean Squared Error of the model:")
        display.write(mean_squared_error(y, pred))
        display.subheader("R Squared score of the model:")
        display.write(r2_score(y, pred))
    else:
        input.write("The selected feature is not in the dataset.")
