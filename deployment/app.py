import streamlit as st
from prediction import run_prediction
from eda import run_eda

# Judul aplikasi
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("Face Detection App")

# Navigasi antar halaman
menu = ["Prediction", "EDA"]
choice = st.sidebar.selectbox("Pilih Halaman", menu)

if choice == "Prediction":
    run_prediction()
elif choice == "EDA":
    run_eda()