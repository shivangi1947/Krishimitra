import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import (
    densenet,
    mobilenet_v2,
    efficientnet
)

# ✅ Streamlit Page Config
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

# Parameters
IMG_SIZE = (128, 128)
CLASS_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy"
]

# Preprocessing
efficientnet_preprocess = efficientnet.preprocess_input
mobilenet_preprocess = mobilenet_v2.preprocess_input
densenet_preprocess = densenet.preprocess_input

@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("efficientnetb0_final.h5")
    model2 = tf.keras.models.load_model("mobilenetv2_finetuned.h5")
    model3 = tf.keras.models.load_model("densenet121_finetuned.h5")
    return model1, model2, model3

model1, model2, model3 = load_models()

def preprocess_image_for_model(image, preprocess_fn):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = preprocess_fn(img_array)
    return np.expand_dims(img_array, axis=0)

def ensemble_predict(image):
    input1 = preprocess_image_for_model(image, efficientnet_preprocess)
    input2 = preprocess_image_for_model(image, mobilenet_preprocess)
    input3 = preprocess_image_for_model(image, densenet_preprocess)

    preds1 = model1.predict(input1, verbose=0)[0]
    preds2 = model2.predict(input2, verbose=0)[0]
    preds3 = model3.predict(input3, verbose=0)[0]

    # Weighted average
    final_pred = (0.03 * preds1 + 0.53 * preds2 + 0.44 * preds3)

    predicted_class = np.argmax(final_pred)
    confidence = final_pred[predicted_class]
    return CLASS_LABELS[predicted_class], confidence, final_pred

# UI

st.markdown("# 🌱 Welcome to the Plant Disease Detection App!")
st.write("This app uses an ensemble of deep learning models to predict the presence of plant diseases.")
st.markdown("---")

# -----------------------
# Supported Species Dropdown
# -----------------------
st.markdown("## 🌾 Supported Species & Diseases")

# Static species and diseases mapping
static_species_diseases = [
    ("🍎 Apple", [
        "Apple scab",
        "Black rot",
        "Cedar apple rust",
        "healthy"
    ]),
    ("🫐 Blueberry", [
        "healthy"
    ]),
    ("🍒 Cherry (including sour)", [
        "Powdery mildew",
        "healthy"
    ]),
    ("🌽 Corn (maize)", [
        "Cercospora leaf spot / Gray leaf spot",
        "Common rust",
        "Northern Leaf Blight",
        "healthy"
    ]),
    ("🍇 Grape", [
        "Black rot",
        "Esca (Black Measles)",
        "Leaf blight (Isariopsis Leaf Spot)",
        "healthy"
    ]),
    ("🍊 Orange", [
        "Haunglongbing (Citrus greening)"
    ]),
    ("🍑 Peach", [
        "Bacterial spot",
        "healthy"
    ]),
    ("🫑 Pepper, bell", [
        "Bacterial spot",
        "healthy"
    ])
]

# Color mapping
color_map = {
    "🍎": "#d62828",
    "🫐": "#4f518c",
    "🍒": "#b5179e",
    "🌽": "#f4a261",
    "🍇": "#6a0572",
    "🍊": "#f77f00",
    "🍑": "#ffb347",
    "🫑": "#2a9d8f"
}

# Two per row display (compact layout + colored list items)
for i in range(0, len(static_species_diseases), 2):
    cols = st.columns(2)
    for col_idx in range(2):
        if i + col_idx < len(static_species_diseases):
            species_name, disease_list = static_species_diseases[i + col_idx]
            emoji = species_name.split(" ")[0]
            color = color_map.get(emoji, "#000000")  # Fallback to black
            with cols[col_idx].expander(species_name):
                for disease in disease_list:
                    st.markdown(
                        f"<li style='margin-left:10px; font-size:15px; color:{color}'>{disease}</li>",
                        unsafe_allow_html=True
                    )

st.markdown("Upload an image of a plant leaf, and this ensemble-powered model will predict the disease class with high accuracy.")

# Upload Box
uploaded_file = st.file_uploader("📤 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="🖼️ Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("🔍 Running ensemble prediction..."):
            predicted_label, confidence, probabilities = ensemble_predict(image)
        
        st.success("✅ Prediction Complete!")
        st.markdown(f"### 🦠 Likely Disease: `{predicted_label}`")

        # Chatbot
        st.markdown(
            """
            <a href="https://549e1cfa6993f02828.gradio.live" target="_blank" style="text-decoration: none;">
                <div style="margin-top: 20px; padding: 10px; border-left: 5px solid #f4a261; background-color: #fff3e0; border-radius: 5px; color: black;">
                    💡 <strong>Get AI assisted help</strong>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

    with st.expander("📊 Show All Class Probabilities", expanded=False):
        prob_df = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Probability": probabilities
        }).set_index("Class").sort_values("Probability", ascending=True)
        st.bar_chart(prob_df)

else:
    # 💡 AI Assistance Chatbot
        st.markdown(
            """
            <a href="https://549e1cfa6993f02828.gradio.live" target="_blank" style="text-decoration: none;">
                <div style="margin-top: 20px; padding: 10px; border-left: 5px solid #f4a261; background-color: #fff3e0; border-radius: 5px; color: black;">
                    💡 <strong>Get AI assisted help</strong>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )


