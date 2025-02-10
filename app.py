#!/usr/bin/env python
# coding: utf-8

# pip install streamlit
# 

# In[11]:


import streamlit as st
import pickle
import numpy as np
import os


st.set_page_config(page_title="Appointment No-Show Prediction", layout="centered")


background_image = "https://www.istockphoto.com/collaboration/boards/_3cFXmYLXEqIvZYu8Swzog"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({background_image});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stTitle {{
        text-align: center;
        font-size: 36px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }}
    .stMarkdown {{
        text-align: center;
        font-size: 18px;
        color: white;
    }}
    .stButton>button {{
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


model_path = "random_forest.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        rfc = pickle.load(model_file)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Error: Model file not found. Train and save 'random_forest.pkl' first.")
    rfc = None


st.markdown("<h1 class='stTitle'>Appointment No-Show Prediction</h1>", unsafe_allow_html=True)


age = st.number_input("Enter Age:", min_value=0, max_value=100, step=1)

day_of_week = st.selectbox(
    "Select Day of Appointment:",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)


days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_encoding = [1 if day_of_week == d else 0 for d in days]

scholarship = st.radio("Has Scholarship?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
diabetes = st.radio("Has Diabetes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alcoholism = st.radio("Has Alcoholism?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
sms_received = st.radio("Received SMS Reminder?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
hypertension = st.radio("Has Hypertension?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
handicap = st.radio("Has Handicap?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")


gender = st.radio("Select Gender:", ["Male", "Female"])
gender_encoding = [1, 0] if gender == "Female" else [0, 1]  # Gender_F, Gender_M


user_input = np.array(
    [age] + day_encoding + [scholarship, diabetes, alcoholism, sms_received, hypertension, handicap] + gender_encoding
).reshape(1, -1)


st.write(f"Feature vector shape: {user_input.shape}")


if st.button("Predict"):
    if rfc:
        prediction = rfc.predict(user_input)
        if prediction[0] == 1:
            st.success("✅ Prediction: The patient is likely to **show up** for the appointment.")
        else:
            st.warning("❌ Prediction: The patient is likely to **miss the appointment** (No-Show).")
    else:
        st.error("❌ Model not loaded. Check if 'random_forest.pkl' exists.")


# In[ ]:




