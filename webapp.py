
from cv2 import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import streamlit as st
import os

model = tf.keras.models.load_model('model_VGG16.hdf5')

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

st.write("""
         # Melanoma Prediction
         """
         )
st.write("This is a simple image classification web app to predict whether a skin lesion is cancerous")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(150, 150),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) < 0.5:
        st.write("Melanoma")
    elif np.argmax(prediction) > 0.5:
        st.write("Non-cancerous")
    
    
    st.text("Probability (0: Melanoma, 1: Non-cancerous)")
    st.write(prediction)

#run in terminal using: streamlit run <filename>
