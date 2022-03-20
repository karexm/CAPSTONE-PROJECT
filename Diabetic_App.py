#load the libraries
from typing import TypeVar, Text
from multimethod import RETURN
AnyStr = TypeVar('AnyStr', Text, bytes)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

#create the title and the subtitle
st.title("DIABETIC PREDICTION APP")
st.write(""" #Using diabetes data to determine whether an individual has diabetes or not!""")

#open and display the data
image=Image.open("C:/Users/HP/Documents/python files/diabetes-test.jpg ")
st.image(image,caption= "Blood sugar screening test",use_column_width=True)

#load the data
df=pd.read_csv("C:/Users/HP/Documents/python files/diabetes.csv")

st.subheader("DATA INFORMATION:")
#show the data as a table
st.dataframe(df)


st.subheader("DATA DESCRIPTION:")
#show statistics of the data /describe data
st.write(df.describe())


st.subheader("DATA VISUALIZATION:")
chart=st.bar_chart(df)


#split the data into feature X and label Y
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


#create and train a  model
clf=RandomForestClassifier()
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)


#show the model metrics
st.subheader("MODEL TEST ACCURACY:")
st.write(str(accuracy_score(Y_test,clf.predict(X_test))*100)+"%" )


#store the model predictions in a variable
prediction=clf.predict(X)



#set a subheader and display the classification
st.subheader("CLASSIFICATION:")
st.write(prediction)


st.sidebar.header("DIABETES PARAMETERS")
#get features input from the user
# def get_user_input():
Pregnancies = st.sidebar.slider("Pregnancies",0,14,2)
PlasmaGlucose= st.sidebar.slider("PlasmaGlucose",44,192,104) 
DiastolicBloodPressure = st.sidebar.slider("DiastolicBloodPressure",24,117,72)
TricepsThickness =st.sidebar.slider("TricepsThickness",7,93,31)
SerumInsulin= st.sidebar.slider("SerumInsulin",14,799,83)
BMI = st.sidebar.slider("BMI",18,56,32)
DiabetesPedigree= st.sidebar.slider("DiabetesPedigree",0.0780,2.3016,0.2003)
Age = st.sidebar.slider("Age",21,77,24)


#make a dictionary to store variables
user_data={"Pregnancies":Pregnancies,
"PlasmaGlucose": PlasmaGlucose,
"DiastolicBloodPressure": DiastolicBloodPressure,
"TricepsThickness": TricepsThickness,
"SerumInsulin": SerumInsulin,
" BMI":  BMI,
"DiabetesPedigree": DiabetesPedigree,
"Age": Age}

#transform the feature into a dataframe
features=pd.DataFrame(user_data,index=[0])
features.head()


#store the users_input into a variable
user_input=user_data

#set a sub header and display the users
st.subheader("USER INPUT:")
st.write(user_input)



st.title("PREDICTION")
#creating a function for prediction
def diabetes_prediction(user_data):

#changing the input_data to numpy array
  user_data_as_numpy_array = np.as_array(user_data)
#reshape the  user data array
  user_data_reshaped=user_data_as_numpy_array.reshaped(1,-1)
  user_data=user_data_reshaped
  prediction=clf.predict(user_data)
  prediction

if(prediction[0]== 0):
    "The person is not diabetic"
else:
    "The person is diabetic"


