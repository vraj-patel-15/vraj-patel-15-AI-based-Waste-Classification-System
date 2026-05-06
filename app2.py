import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Waste Classifier", page_icon="♻️")

# --- UI Header ---
st.title("♻️ AI Waste Classifier")
st.markdown("Upload a photo of waste (plastic, paper, etc.)")


# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        # Ensure this filename matches your saved model file
        return tf.keras.models.load_model('model.keras')
    except Exception as e:
        st.error(f"Could not find model file. Error: {e}")
        return None


model = load_model()
MY_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Mapping Logic & Tips
RECYCLE_LOGIC = {
    'cardboard': {'cat': 'Recyclable', 'tip': 'Flatten boxes to save space.'},
    'glass': {'cat': 'Recyclable', 'tip': 'Rinse out food residue; labels are okay.'},
    'metal': {'cat': 'Recyclable', 'tip': 'Empty and rinse aluminum/steel cans.'},
    'paper': {'cat': 'Recyclable', 'tip': 'Keep it dry! Wet or greasy paper cannot be recycled.'},
    'plastic': {'cat': 'Recyclable', 'tip': 'Check the number on the bottom (usually #1 and #2 are best).'},
    'trash': {'cat': 'Non-Recyclable', 'tip': 'Dispose of in general waste bin.'}
}

# --- Sidebar ---
st.sidebar.title("Model Details")
st.sidebar.info("Base Model: MobileNetV2\nClasses: 6\nInput Size: 224x224")

# --- Image Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # 1. Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # 2. Preprocessing
    with st.spinner("AI is analyzing material..."):
        if image.mode != "RGB":
            image = image.convert("RGB")

        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Predict
        predictions = model.predict(img_array)
        score = predictions[0]
        class_idx = np.argmax(score)
        label = MY_CLASSES[class_idx]
        confidence = score[class_idx] * 100

    # 4. Display Results
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Material: {label.upper()}")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    with col2:
        category = RECYCLE_LOGIC[label]['cat']
        tip = RECYCLE_LOGIC[label]['tip']

        if category == 'Recyclable':
            st.success(f"Category: **{category}** ♻️")
        else:
            st.error(f"Category: **{category}** 🗑️")

        st.info(f"**Pro Tip:** {tip}")

    # Bar Chart for PPT visualization
    st.write("### Probability Breakdown")
    chart_data = {MY_CLASSES[i]: float(score[i]) for i in range(len(MY_CLASSES))}
    st.bar_chart(chart_data)

elif model is None:
    st.warning("Please ensure 'waste_classifier_model.h5' is in the same folder as this script.")
