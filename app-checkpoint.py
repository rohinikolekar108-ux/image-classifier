import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}
.result {
    font-size: 28px;
    font-weight: bold;
    color: #27ae60;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = load_model("fruit_classifier.h5")

class_names = ['apple', 'grape', 'orange', 'pineapple', 'strawberry']

# ------------------ UI Header ------------------
st.markdown('<div class="title">🍓 Fruit Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered fruit recognition using CNN</div>', unsafe_allow_html=True)

# ------------------ Upload Section ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction ------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict Fruit"):
        prediction = model.predict(img_array)[0]
        index = np.argmax(prediction)
        result = class_names[index]
        confidence = prediction[index] * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result">🍎 Prediction: {result.upper()}</div>', unsafe_allow_html=True)
        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<center>💻 Developed by Rohini | AI Fruit Classification Project</center>",
    unsafe_allow_html=True
)
