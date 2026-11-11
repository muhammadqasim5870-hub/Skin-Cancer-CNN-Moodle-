
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Melanoma Skin Cancer Predictor",
    page_icon="🔬",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "melanoma_cnn_model.keras"
IMG_SIZE = (224, 224)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("CNN model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# App title
st.title("🔬 Melanoma Skin Cancer Predictor")
st.write(
    "Upload a skin image to see the CNN model's prediction."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        # Preprocess image
        resized_image = image.resize(IMG_SIZE)

        image_array = np.array(
            resized_image,
            dtype=np.float32
        ) / 255.0

        image_array = np.expand_dims(
            image_array, axis=0
        )

        # Prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        probability = float(np.array(prediction).squeeze())

        # Class mapping: 0 = Benign, 1 = Malignant
        if probability >= 0.5:
            predicted_class = "Malignant"
            confidence = probability * 100
        else:
            predicted_class = "Benign"
            confidence = (1 - probability) * 100

        st.subheader("Prediction Result")

        if predicted_class == "Malignant":
            st.error(f"Prediction: {predicted_class}")
        else:
            st.success(f"Prediction: {predicted_class}")

        st.write(f"Model confidence: {confidence:.2f}%")

        st.warning(
            "This is an educational AI model, not a medical "
            "diagnosis. A qualified dermatologist must assess "
            "any suspicious skin lesion."
        )
