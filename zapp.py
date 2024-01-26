import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background
set_background('brain.png')
# Set page title
st.title("Brain Stroke Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the model
model = load_model("brainstroke.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Display image and classify
if uploaded_file is not None:
    # Preprocess the image
    img1 = Image.open(uploaded_file)
    img = img1.resize((64, 64))  # Assuming your model expects 64x64 images
    img_array = np.array(img)
    img_array = img_array.reshape(1, 64, 64, 3)
    img_array = img_array.astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display results
    st.image(img1, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:")
    st.write(f"Class: {class_name[2:]}")
    st.write(f"Confidence: {confidence_score:.2%}")
