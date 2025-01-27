import streamlit as st
import pandas as pd
import numpy as np
import joblib 

st.title("Insurance Cost Prediction")

model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

col1, col2 = st.columns(2)

age = col1.slider("Set the Age", 18, 66, step=1)
height = col1.slider("Set the Height", 145, 188, step=1)
weight = col1.slider("Set the Weight", 51, 132, step=1)
bmi = weight/((height/100)**2)
no_of_surgeries = col1.slider("Set the No. of major Surgeries", 0, 3, step=1)

diabetes = col2.selectbox("Person has diabetes",["Yes", "No"])
bp = col2.selectbox("Person has blood pressure-related issues",["Yes", "No"])
transplant = col2.selectbox("Person has had a transplant.",["Yes", "No"])
chronic_disease = col2.selectbox("Person has any chronic diseases",["Yes", "No"])
allergies = col2.selectbox("Person has any known allergies",["Yes", "No"])
cancer = col2.selectbox("Person has family history of cancer",["Yes", "No"])

encode_dict = {
    "diabetes": {'Yes': 1, 'No': 0},
    "bp": {'Yes': 1, 'No': 0},
    "transplant": {'Yes': 1, 'No': 0},
    "chronic_disease": {'Yes': 1, 'No': 0},
    "allergies": {'Yes': 1, 'No': 0},
    "cancer": {'Yes': 1, 'No': 0}
}

if st.button("Get Premium"):
    diabetes = encode_dict['diabetes'][diabetes]
    bp = encode_dict['bp'][bp]
    transplant = encode_dict['transplant'][transplant]
    chronic_disease = encode_dict['chronic_disease'][chronic_disease]
    allergies = encode_dict['allergies'][allergies]
    cancer = encode_dict['cancer'][cancer]

    numerical_features = ['Age', 'Height', 'Weight', 'bmi']
    scaled_cols = scaler.transform(np.array([age,height,weight,bmi]).reshape(-1, 4))
    all_cols = np.concatenate((scaled_cols, diabetes, bp, transplant, chronic_disease, allergies, cancer, no_of_surgeries), axis=None).reshape(-1,11)

    y_pred = model.predict(all_cols)[0]
    st.header(y_pred)



