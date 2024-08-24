import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

#Heading...
st.header("Welcome to Data Scientist, Samad Platform!")
st.write("---")
image1=Image.open("pic.jpg")
st.image(image1)

st.write("---")
st.write("""


        ## Explore Different ML models and datasets
         Which Model is best?
         """)

#datasets 

dataset_name=st.sidebar.selectbox('Select Dataset',options=('Iris','Breast_Cancer','Wine'))

#Classifier name
classifier_name=st.sidebar.selectbox('Select Classifier',('KNN','SVM','Random Forest'))

#Define a Function
def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=='Wine':
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    x=data.data
    y=data.target
    return x,y

X,y=get_dataset(dataset_name)

st.write("Shape of data",X.shape)
st.write("Number of Classes: ",len(np.unique(y)))

#Add the input user parameter in different classifier
def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators


    return params

params=add_parameter_ui(classifier_name)

#Make the classifier

def get_classifier(classifier_name,params):
    clf=None
    if classifier_name =='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name=='SVM':
        clf=SVC(C=params['C'])
    else:
        clf=RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)

#split the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=1234)

#Train the model
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#Accuracy Score of model
acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier: {classifier_name}')
st.write(f'Accuracy: {acc}')

#Plot dataset
pca=PCA(2)
x_projected=pca.fit_transform(X)

#Slicing the data into 0 or 1 dimension

x1=x_projected[:,0]
x2=x_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,
            c=y,alpha=0.8,
            cmap='viridis')
st.pyplot(fig)

st.write("---")

st.write("### Ab agr ap song ko enjoy krna chahty hn tu yeh sunein:")
audio1=open("Bahut Pyaar Karte Hai.mp3",'rb')
st.audio(audio1)

st.write("---")
