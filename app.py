import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model(r"C:\Users\Rohini\Desktop\Fruit_classifier\fruit_classifier.h5")

class_names = ['apple', 'grape', 'orange', 'pineapple', 'strawberry']

st.set_page_config(page_title="Fruit Classifier 🍎", layout="centered")

st.title("🍓 Fruit Classification App")
st.write("Upload a fruit image and get prediction")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((64, 64))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = class_names[np.argmax(prediction)]

    st.success(f"🍉 Predicted Fruit: **{result.upper()}**")
