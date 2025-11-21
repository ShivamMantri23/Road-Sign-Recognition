
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load trained model
model = load_model('GTSRB_HighAccuracy.h5')


st.title("🚦 GTSRB Road Sign Recognition – Live Demo")
st.write("Upload a traffic sign image to classify it.")


uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","ppm"])


if uploaded:
   img = Image.open(uploaded).convert('RGB')
   st.image(img, caption="Uploaded Image", use_column_width=True)


   img_resized = img.resize((32,32))
   arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)


   pred = model.predict(arr)
   cls = int(np.argmax(pred))
   conf = float(pred[0][cls])


   st.success(f"Predicted Class: {cls}")
   st.info(f"Confidence: {conf:.4f}")


