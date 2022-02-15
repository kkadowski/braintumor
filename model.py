import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import datetime
from PIL import Image

st.set_page_config(page_title="K>K>",initial_sidebar_state="expanded")
st.caption('*Brain tumor* prediction')

st.sidebar.title('Welcome')
class_names= ['Tumor', 'Healhty']
file = st.sidebar.file_uploader("Choose a photo ", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=False)
if file is not None:
    image = Image.open(file)
    try:
      new_model = tf.keras.models.load_model('model.h5')
    except OSError:
      "Can''t read model..."
    else:
      img_array = np.array(image)
      img = tf.image.resize(img_array, size=(244,244))
      img = tf.expand_dims(img, axis=0)
      pred = new_model.predict(img)
      st.image(
        image,
        caption=f"Photo calssification: {class_names[np.argmax(pred)]}",
        use_column_width=True,
    )