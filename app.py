import streamlit as st
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (224, 224)  # change if your model uses different input size
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")  # <-- your model file
    return model

model = load_model()

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# UI
# ----------------------------
st.title("🗑 Waste Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[class_index]}")
    st.write(f"**Confidence:** {confidence:.2f}")

# ----------------------------
# CONFUSION MATRIX SECTION
# ----------------------------
st.header("📊 Confusion Matrix (Validation Set)")

if st.button("Generate Confusion Matrix"):

    y_true = []
    y_pred = []

    for images, labels in val_ds:   # IMPORTANT: you must define val_ds
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Heatmap")

    st.pyplot(fig)