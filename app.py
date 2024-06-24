import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('C:\ML\dogs vs cats.h5')

# Function to predict the image
def predict_image(model, img):
    img = img.resize((150, 150))  # Resize to the input size of the model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return 'Dog' if prediction[0] > 0.5 else 'Cat'

# Streamlit app
st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_image(model, img)
    st.write(f'This is a {label}.')
