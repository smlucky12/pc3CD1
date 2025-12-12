import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Cargar modelo
def load_flower_model():
    model = load_model('best_tuned_model.h5')
    return model

model = load_flower_model()

# Configuración de la página
st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸", layout="centered")

st.title("🌼 Clasificador de Flores con CNN")
st.write("Sube una imagen de una flor y el modelo la clasificará.")

# Subida de imagen
uploaded_img = st.file_uploader("Sube una imagen de una flor", type=["jpg", "jpeg", "png"])

# Mapeo de clases (editar según las clases reales)
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img, caption="Imagen subida", width=300)

    # Preprocesamiento
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predicción
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    confidence = np.max(prediction)

    st.success(f"🌸 Predicción: **{class_name.capitalize()}**")
    st.info(f"Confianza: {confidence:.2f}")

st.write("---")
st.write("Desarrollado con ❤️ usando Streamlit y Redes Neuronales Convolucionales.")
