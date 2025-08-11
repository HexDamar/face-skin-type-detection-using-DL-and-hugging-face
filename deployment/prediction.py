# prediction.py

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Load model dan class names
model = load_model('improved_model.keras')
class_names = ['dry', 'normal', 'oily',]

def preprocess_image(image, img_height=224, img_width=224):
    img = load_img(image, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def run_prediction():
    st.header("Prediksi Wajah Anda Mirip Siapa")

    uploaded_file = st.file_uploader("Upload gambar wajah kamu (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        x, img_display = preprocess_image(uploaded_file)

        # Prediksi
        y_pred_prob = model.predict(x)
        y_pred_class = np.argmax(y_pred_prob)
        pred_class_name = class_names[y_pred_class]

        # Tampilkan gambar
        st.image(img_display, caption=f"Gambar yang diunggah", use_column_width=True)

        # Hasil prediksi
        st.subheader(f"Hasil Prediksi: {pred_class_name}")
        st.write("Persentase kemiripan ke tiap kelas:")

        # Tampilkan semua probabilitas
        for i, prob in enumerate(y_pred_prob[0]):
            st.write(f"- {class_names[i]}: {prob * 100:.2f}%")