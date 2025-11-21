import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

# ----------------------------------------------------
# 1. Load the trained model
# ----------------------------------------------------
MODEL_PATH = "final_clothing_classifier.keras"
model = load_model(MODEL_PATH)

# ----------------------------------------------------
# 2. Class names (IMPORTANT — paste your class list)
# ----------------------------------------------------
# SAME ORDER as train_gen.class_indices.keys()

class_names = [
    "Blazer", "Blouse", "Body", "Dress", "Hat", 
    "Hoodie", "Longsleeve", "Other", "Outwear",
    "Pants", "Polo", "Shirt", "Shoes", "Shorts",
    "Skip", "Skirt", "T-Shirt", "Top", "Undershirt"
]

IMG_SIZE = 224

# ----------------------------------------------------
# 3. Preprocess function (ResNet50 pipeline)
# ----------------------------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    if img.shape[-1] == 4:   # Convert RGBA → RGB
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# ----------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------
st.title("👕 Clothing Image Classifier")
st.write("Upload an image to predict the clothing category.")

uploaded_file = st.file_uploader("Upload clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed = preprocess_image(img)

    # Predict
    preds = model.predict(processed)[0]     # shape: (19,)
    top3_idx = preds.argsort()[::-1][:3]    # highest 3

    st.subheader("Top 3 Predictions:")
    for i in top3_idx:
        st.write(f"**{class_names[i]}** — {preds[i]*100:.2f}%")

    # Optional: Progress bars
    st.subheader("Confidence Scores:")
    for i in top3_idx:
        st.write(f"{class_names[i]}:")
        st.progress(float(preds[i]))
