import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

model_path=r"C:\Users\USER\OneDrive\Desktop\Python\fashion\fashionClassifierModel.h5"

model=tf.keras.models.load_model(model_path)


class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


def process_image(img):
    img=img.convert('L')   #'L'->Grayscale (0–255) , RGB->'Red, Green, Blue, 'RGBA'->RGB + Alpha (transparency) ,1->Black & White (binary),P->Palette-based (indexed colors)
    img=img.resize((28,28))
    img_array=np.array(img)
    img_array=img_array/255.0
    img_array=img_array.reshape((1,28,28,1))
    return img_array


st.title("Hey You Know Your Fashion Name ")

upload_image=st.file_uploader("Upload an Image ..",type=["jpg","jpeg","png"])

if upload_image is not None:
    img=Image.open(upload_image)    #PIL Image object, NOT an array
    col1,col2=st.columns(2)

    with col1:
        resized_img=img.resize((100,100))
        st.image(resized_img)
    
    with col2:
        if st.button("Classify.."):
            img_array=process_image(img)
            result=model.predict(img_array)
            st.write(str(result))

            predicted_class=np.argmax(result) #it return 10 value argmax help to find the index where max value 

            prediction=class_names[predicted_class]

            st.success(f"Predicted: {prediction}")







st.header("letssss Goooo")

    

