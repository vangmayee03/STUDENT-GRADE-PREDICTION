import streamlit as st
import pandas as pd
import joblib
import os
import streamlit.web.cli as stcli
import sys

# Load model
@st.cache_resource
def load_model():
    return joblib.load("student_grade_pipeline.pkl")

model = load_model()

# Title
st.title("Student Grade Prediction")

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Prep Course", ["none", "completed"])
math = st.slider("Math Score", 0, 100, 70)
reading = st.slider("Reading Score", 0, 100, 70)
writing = st.slider("Writing Score", 0, 100, 70)

# Submit and predict
if st.button("Predict Grade"):
    data = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": prep,
        "math score": math,
        "reading score": reading,
        "writing score": writing
    }])

    prediction = model.predict(data)[0]
    st.success(f"Predicted Grade Category: **{prediction}**")

if __name__ == "__main__" and "RENDER" in os.environ:
    sys.argv = [
        "streamlit", "run", "app.py",
        "--server.port=" + str(os.environ.get("PORT", 8501)),
        "--server.address=0.0.0.0"
    ]
    sys.exit(stcli.main())
