import streamlit as st

st.title(
    "Radiology Disease Prediction AI"
)

report = st.text_area(
    "Paste Radiology Report"
)

if st.button("Predict"):

    st.success(
        "Prediction system connected"
    )