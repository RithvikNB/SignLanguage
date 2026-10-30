import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Page setup
st.set_page_config(
    page_title="ASL Sign Language Classifier",
    layout="wide"
)

st.title(" ASL Sign Language Classifier")

@st.cache_resource
def load_model():
    model_path = "asl_mobilenetv2.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in {os.getcwd()}")
        return None
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_labels():
    labels_path = "labels.txt"
    if not os.path.exists(labels_path):
        st.error(f"Labels file '{labels_path}' not found in {os.getcwd()}")
        return []
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

model = load_model()
labels = load_labels()

# Sidebar controls for fine-tuning
st.sidebar.header("Model & Crop Settings")

norm_mode = st.sidebar.selectbox(
    "Pixel Normalization (Match train.py):",
    ("Raw [0, 255]", "Scaled [0, 1]", "MobileNetV2 [-1, 1]")
)

crop_percent = st.sidebar.slider("Crop Box Size (%)", min_value=30, max_value=80, value=45, step=5) / 100.0
y_offset = st.sidebar.slider("Vertical Shift (Y-Axis)", min_value=-30, max_value=30, value=-10, step=5) / 100.0

def crop_hand_region(image_np, crop_pct, y_off):
    """Crops closely around the center hand region."""
    h, w, _ = image_np.shape
    crop_dim = int(min(h, w) * crop_pct)
    
    start_x = max(0, (w - crop_dim) // 2)
    start_y = max(0, int((h - crop_dim) // 2 + (h * y_off)))
    
    end_x = min(w, start_x + crop_dim)
    end_y = min(h, start_y + crop_dim)
    
    return image_np[start_y:end_y, start_x:end_x]

# Main Interface
option = st.radio("Select Input Source:", ("Upload Image", "Webcam Capture"))

image_file = None
if option == "Upload Image":
    image_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo of an ASL sign")

if image_file is not None and model is not None:
    pil_img = Image.open(image_file).convert("RGB")
    img_np = np.array(pil_img)

    # Apply crop
    if option == "Webcam Capture":
        processed_img = crop_hand_region(img_np, crop_percent, y_offset)
    else:
        processed_img = img_np

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original Input", use_container_width=True)
    with col2:
        st.image(processed_img, caption="Cropped Focus (Fed to Model)", use_container_width=True)

    # Preprocessing
    img_resized = cv2.resize(processed_img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)

    if norm_mode == "Scaled [0, 1]":
        img_array = img_array / 255.0
    elif norm_mode == "MobileNetV2 [-1, 1]":
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Inference
    with st.spinner("Classifying gesture..."):
        predictions = model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx] * 100
        predicted_label = labels[top_idx] if top_idx < len(labels) else f"Class {top_idx}"

    st.markdown("---")
    st.subheader(f"Prediction: **{predicted_label}**")
    st.progress(float(predictions[top_idx]))
    st.write(f"**Confidence:** {confidence:.2f}%")

    if len(labels) > 1:
        st.subheader("Top Match Breakdown")
        top_k = min(5, len(predictions))
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        for idx in top_indices:
            label_name = labels[idx] if idx < len(labels) else f"Class {idx}"
            prob = predictions[idx]
            st.write(f"**{label_name}**: {prob * 100:.1f}%")
            st.progress(float(prob))
