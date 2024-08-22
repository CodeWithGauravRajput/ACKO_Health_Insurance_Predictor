import streamlit as st
import joblib
import numpy as np

# Page Title and Logo
st.image("logo.png", width=150)
st.title("Acko Health Insurance Predictor App")

# Banner Image
st.image("banner.jpg", use_column_width=True)

age = st.number_input("Enter your Age", min_value=10, max_value=90, value=30, step=1)
height = st.number_input("Enter your Height in meters", min_value=0.6, max_value=2.7, value=1.67)
weight = st.number_input("Enter your Weight in kg", min_value=25, max_value=200, value=80, step=1)
children = st.number_input("Enter number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox('Do you Smoke?', ('Yes', 'No'))

bmi = weight / (height) ** 2
smoker_num = 0 if smoker == 'No' else 1
test_data = [[age, bmi, children, smoker_num]]

# Display message for smokers
if smoker == 'Yes':
    st.warning("mat piya ker bhai")

# Model load
model = joblib.load("insurance_joblib")
poly = joblib.load("poly_obj")

if st.button('Get Quote'):
    test_poly = poly.transform(test_data)
    y_pred_log = model.predict(test_poly)
    premium = round(np.exp(y_pred_log)[0], 2)
    st.write(f'## **Your Premium amount is $ {premium}**')
