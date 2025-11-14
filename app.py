# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Tree Species Classifier", layout="centered")

# ---------------------------
# CONFIG - using your provided model filename
# ---------------------------
MODEL_PATH = "basic_cnn_tree_species.h5"   # <<-- your uploaded model
CLASSES_PATH = "classes.json"
IMG_SIZE = (224, 224)  # keep as-is; if you trained with a different size, change here

# ---------------------------
# LOAD MODEL + CLASSES
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    # Load model
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Load class labels if available
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            idx2class = json.load(f)
        # ensure int keys
        idx2class = {int(k): v for k, v in idx2class.items()}
    else:
        # fallback labels (0..N-1)
        num_classes = model.output_shape[-1]
        idx2class = {i: f"Class {i}" for i in range(num_classes)}
    
    return model, idx2class

model, idx2class = load_model_and_classes()

st.title("🌳 Tree Species Classifier")
st.write("Upload an image of a tree/leaf/bark — the model will predict the species.")

# ---------------------------
# IMAGE UPLOADER
# ---------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Please upload an image to begin.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)

# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess_image(pil_img, target_size):
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
        st.stop()

    x = preprocess_image(img, IMG_SIZE)
    preds = model.predict(x)
    probs = preds[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100
    label = idx2class.get(top_idx, f"Class {top_idx}")

    st.success(f"**Prediction: {label}** ({confidence:.2f}% confidence)")

    # Show probabilities table
    st.subheader("Class Probabilities")
    sorted_indices = np.argsort(probs)[::-1]
    results = [(idx2class.get(int(i), f"Class {i}"), float(probs[i])) 
               for i in sorted_indices]

    st.table({
        "Class": [r[0] for r in results],
        "Probability": [round(r[1], 4) for r in results]
    })
