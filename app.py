import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Classificador de Dígitos – CNN (MNIST)")
st.write("Envie uma imagem 28x28 em tons de cinza.")


from tensorflow.keras.activations import softmax
custom_objects = {'softmax_v2': softmax}


model = tf.keras.models.load_model("final_CNN_model.h5", custom_objects=custom_objects)

uploaded_file = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(img, caption="Imagem enviada", width=150)

    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    pred = model.predict(img_array)
    digit = np.argmax(pred)

    st.subheader(f"🔢 Dígito previsto: **{digit}**")

