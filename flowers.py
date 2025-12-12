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

# --- Estilos personalizados con tema floral ---
st.markdown(
    """
    <style>
    body {
        background-color: #fff7fb;
        background-image: url('https://e0.pxfuel.com/wallpapers/85/238/desktop-wallpaper-flowers-aesthetic-and-flower-blue-flowers-aesthetic.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.65);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(6px);
    }
    h1 {
        color: #d63384;
        text-shadow: 1px 1px 2px #ffb3d9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌼 Clasificador de Flores con CNN")
st.write("Sube una imagen de una flor y el modelo la clasificará.")

# Subida de imagen
uploaded_img = st.file_uploader("Sube una imagen de una flor", type=["jpg", "jpeg", "png"])

# Mapeo de clases (editar según las clases reales)
# Mapeo de clases en español
class_names = ["Margarita", "Diente de león", "Rosa", "Girasol", "Tulipán"]

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

    st.success(f"🌸 Predicción: **{class_name}**")
    st.info(f"Nivel de confianza: {confidence:.2f}")

st.write("---")
st.write("Desarrollado con ❤️ usando Streamlit y Redes Neuronales Convolucionales.")
