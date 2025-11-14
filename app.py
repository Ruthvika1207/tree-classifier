# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Tree Species Classifier", layout="centered")

# ---------- CONFIG ----------
MODEL_PATH = "basic_cnn_tree_species.h5"
CLASSES_PATH = "classes.json"
IMG_SIZE = (224, 224)  # change if necessary

# ---------- UTIL ----------
def preprocess_image(pil_img, target_size):
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype("float32")
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- LOAD ----------
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            raw = json.load(f)
        idx2class = {int(k): v for k, v in raw.items()}
    else:
        out_shape = model.output_shape
        if len(out_shape) >= 2 and out_shape[-1] == 1:
            idx2class = {0: "Class 0", 1: "Class 1"}
        else:
            num = out_shape[-1]
            idx2class = {i: f"Class {i}" for i in range(num)}
    return model, idx2class

model, idx2class = load_model_and_classes()

st.title("🌳 Tree Species Classifier")
st.write("Upload an image and click Predict. Supports binary and multi-class models.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Please upload an image to begin.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)

# Debug toggle: enable to print more internals for troubleshooting
debug = st.checkbox("Show debug info", value=False)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
        st.stop()

    x = preprocess_image(img, IMG_SIZE)

    # When debugging, show preprocessing & input tensor info
    if debug:
        st.subheader("Debug: input / preprocessing")
        st.write({
            "PIL mode": getattr(img, "mode", None),
            "PIL size": getattr(img, "size", None),
            "preprocessed_shape": x.shape,
            "dtype": str(x.dtype)
        })
        try:
            st.write("pixel range / stats:", {
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean())
            })
            flat = x.flatten()
            st.write("first 20 preprocessed values:", flat[:20].tolist())
        except Exception as e:
            st.write("Could not compute stats:", e)

    try:
        preds = model.predict(x)
        out = preds[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Raw prediction output
    st.subheader("Raw model output (first 10 values)")
    try:
        # out might be scalar-like or an array
        preview = out[:10].tolist() if hasattr(out, "size") and out.size >= 10 else np.atleast_1d(out).tolist()
        st.write(preview)
    except Exception:
        st.write(str(out))
    st.write("Output shape:", np.atleast_1d(out).shape)

    # Extra debugging information about model and prediction
    if debug:
        st.subheader("Debug: model and prediction internals")
        try:
            st.write({
                "model_input_shape": getattr(model, "input_shape", None),
                "model_output_shape": getattr(model, "output_shape", None)
            })
            st.text("Model summary:")
            try:
                model.summary(print_fn=lambda s: st.text(s))
            except Exception as e:
                st.text(f"(could not print model.summary: {e})")
        except Exception as e:
            st.write("Could not inspect model:", e)

        # Show raw preds array and basic checks
        try:
            st.write("Full raw preds:", preds.tolist() if hasattr(preds, "tolist") else str(preds))
            st.write("preds dtype/shape:", getattr(preds, "dtype", None), np.shape(preds))
            st.write("Any NaN/inf in preds:", bool(np.isnan(preds).any()), bool(np.isinf(preds).any()))
        except Exception as e:
            st.write("Could not show preds details:", e)

        # Show classes mapping loaded earlier
        try:
            st.write("Loaded class mapping (idx2class):", idx2class)
        except Exception:
            st.write("No class mapping available")

        # Quick self-tests: run model on zero and random inputs to detect constant output
        try:
            st.write("Running quick model self-test (zero / random inputs)...")
            zero = np.zeros_like(x)
            rand = np.random.random_sample(x.shape).astype(x.dtype)
            batch = np.concatenate([x, zero, rand], axis=0)
            test_preds = model.predict(batch)
            st.write("self-test preds:", test_preds.tolist() if hasattr(test_preds, "tolist") else str(test_preds))
            # check if outputs are identical across these three
            try:
                first = np.array(test_preds[0]).ravel()
                second = np.array(test_preds[1]).ravel()
                third = np.array(test_preds[2]).ravel()
                identical_01 = np.allclose(first, second)
                identical_02 = np.allclose(first, third)
                st.write({
                    "identical(original,zero)": bool(identical_01),
                    "identical(original,random)": bool(identical_02)
                })
                if identical_01 or identical_02:
                    st.warning("Model produced identical outputs for different inputs — this suggests the trained model may be trivial or there's an input/preprocessing mismatch.")
            except Exception:
                pass
        except Exception as e:
            st.write("Self-test failed:", e)

    # Binary (single output)
    if out.shape[-1] == 1:
        val = float(out[0])
        if 0.0 <= val <= 1.0:
            prob_pos = val
        else:
            prob_pos = float(tf.nn.sigmoid(val).numpy())
        label0 = idx2class.get(0, "Class 0")
        label1 = idx2class.get(1, "Class 1")
        if prob_pos >= 0.5:
            pred_label = label1
            conf = prob_pos * 100.0
        else:
            pred_label = label0
            conf = (1.0 - prob_pos) * 100.0
        st.success(f"**Prediction: {pred_label}** ({conf:.2f}% confidence)")
        st.subheader("Class probabilities (binary)")
        st.table({
            "Class": [label0, label1],
            "Probability": [round(1.0 - prob_pos, 4), round(prob_pos, 4)]
        })
    else:
        probs = tf.nn.softmax(out).numpy()
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = idx2class.get(top_idx, f"Class {top_idx}")
        st.success(f"**Prediction: {label}** ({top_prob*100:.2f}% confidence)")
        st.subheader("Class Probabilities")
        sorted_indices = probs.argsort()[::-1]
        results = [(idx2class.get(int(i), f"Class {i}"), float(probs[i])) for i in sorted_indices]
        st.table({
            "Class": [r[0] for r in results],
            "Probability": [round(r[1], 4) for r in results]
        })
