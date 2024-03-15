import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page title
st.title('Heart Disease Prediction App')

# Input fields
age = st.number_input('Age', value=25, step=1)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=range(4), format_func=lambda x: f'Type {x}')
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', value=120, step=1)
chol = st.number_input('Serum Cholestoral (mg/dl)', value=197, step=1)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
restecg = st.selectbox('Resting Electrocardiographic Results', options=range(3))
thalach = st.number_input('Maximum Heart Rate Achieved', value=150, step=1)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.number_input('ST depression induced by exercise relative to rest', value=1.0, step=0.1)
slope = st.selectbox('The slope of the peak exercise ST segment', options=range(3))
ca = st.slider('Number of major vessels (0-3) colored by flourosopy', min_value=0, max_value=3, value=0)
thal = st.selectbox('Thalassemia', options=range(1, 4))

# Predict button
if st.button('Predict Heart Disease'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_df = pd.DataFrame(input_data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Predict
    prediction = model.predict(input_df)
    
    # Display prediction
    st.subheader('Prediction:')
    result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'
    st.write(result)
