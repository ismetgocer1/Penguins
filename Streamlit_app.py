import streamlit as st
import pandas as pd
import joblib

# Load the saved model
pipeline = joblib.load('pipe_final_RF_model.joblib')

#Add image
st.sidebar.image("penguins.png", width=300, caption="Antartica Penguins") 

# Streamlit title
st.title("Penguin Species Prediction Model")


# User input
culmen_length_mm = st.slider("Culmen Length", 30, 60, 45)
culmen_depth_mm = st.slider("Culmen Depth", 13, 22, 15)
flipper_length_mm = st.slider("Flipper Length", 170, 232, 200)
body_mass_g = st.slider("Body Mass", 2700, 6300, 4000)
sex=st.sidebar.selectbox("Select a Gender", ["Female", "Male"])
island=st.sidebar.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])


# Create a DataFrame from the user input
new_data = pd.DataFrame({
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex],
    'island': [island]})

# Make a prediction
if st.button("Make Prediction"):
    prediction = pipeline.predict(new_data)
    
    # Display the results
    st.write(f"Predicted class: {prediction[0]}")
