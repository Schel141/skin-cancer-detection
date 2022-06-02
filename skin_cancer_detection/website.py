import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as plt
import seaborn as sns
import cv2
import os
import joblib
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Reshape
from tensorflow.keras.metrics import Recall, Precision
import pickle


#[theme]
#base = 'dark'
#primaryColor = 'green'

st.title ("Skin Cancer Detection")

st.write("This website aims to develop a skin cancer detection based on dermatoscopic images and a patient's metadata using a Deep Learning model.")

with st.expander("The Process of Skin Cancer Detection"):
    st.write("The process of skin cancer detection has significantly improved over the last years. Many different")
    st.write(" techniques have been applied. Particularly, the method of image classification has taken the")
    st.write("accuracy of the diagnosis to a whole new level.")
    st.write("...")

col1,col2 = st.columns(2)

with col1:
    st.radio("Choose your Gender", ["male", "female"])
    st.markdown('#')

    st.slider("Choose your Age", 0, 100)
    st.markdown('#')

    location = st.selectbox ("Where is your spot localized?", ['', 'Scalp', 'Ear', 'Face', 'Back', 'Trunk', 'Chest',
       'Upper Extremity', 'Abdomen', 'Lower Extremity',
       'Genital', 'Neck', 'Hand', 'Foot', 'Acral'])
    #with st.container('If Other applies:'):
        #st.write('')

with col2:
    with st.expander("Option 1: Upload an Image"):
        uploaded_file = st.file_uploader("Option 1: Upload your Picture")
        if uploaded_file:
            st.image(uploaded_file, width = 228)

    st.markdown('#')
    st.markdown('#')

    with st.expander("Option 2: Take a Picture"):
        camera_file = st.camera_input ("Take a Picture")
        if camera_file:
            st.image(camera_file, width = 225)

st.markdown('#')

st.checkbox("I agree, that due to limitations we cannot guarantee the complete correchtness of our predicition.")

st.markdown('#')

col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    pass
with col2:
    pass
#with col3:
#    st.button("Submit")
with col4:
    pass
with col5:
    pass


if st.button('Submit'):
    X_input = np.asarray(Image.open(uploaded_file).resize((100,75)))
    X_input_stack = np.stack(X_input)

    #joblib_model = joblib.load('basis_with_aug_model.joblib')
    loaded_model = pickle.load(open('basic_model', 'rb'))
    cancer_type = loaded_model.predict(X_input_stack)
    st.markdown(f'### predicted type: {cancer_type}')

#pickle instead of joblib
