import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import uuid

# -----------------------------
# Función para fondo bonito
# -----------------------------
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url('{url}');
            background-size: cover;
            background-attachment: fixed;
        }}
        .stApp {{
            background: rgba(255, 255, 255, 0.70);
            padding: 20px;
            border-radius: 20px;
            backdrop-filter: blur(6px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Fondo floral elegante
add_bg_from_url("https://marketplace.canva.com/EAGTSMwrgRI/1/0/1600w/canva-fondo-de-pantalla-ordenador-flores-patr%C3%B3n-moderno-bonito-azul-y-amarillo-0i4VVc3TMwg.jpg")

# -----------------------------
# Cargar modelo
# -----------------------------
def load_flower_model():
    model = load_model('best_tuned_model.h5')
    return model

model = load_flower_model()

st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸", layout="centered")

st.title("🌼 Clasificador de Flores con CNN")
st.write("Sube una imagen de una flor y el modelo la clasificará.")



# Subida de imagen
uploaded_img = st.file_uploader("Sube una imagen de una flor", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

# Mapeo de clases
class_names = ["Margarita", "Diente de león", "Rosa", "Girasol", "Tulipán"]

if uploaded_img:
    # Mostrar imagen subida
    img = Image.open(uploaded_img)
    st.image(img, caption="Imagen subida", width=300)

    # Preprocesar
    img_resized = img.resize((150, 150))
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0) / 255.0

    # Predicción
    prediction = model.predict(arr)
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    confidence = np.max(prediction)

    # -----------------------------
    # TARJETA ELEGANTE DEL RESULTADO
    # -----------------------------
    st.markdown(
        f"""
        <div style='background: rgba(255,240,250,0.9); padding:20px; border-radius:20px; text-align:center; 
                    border:2px solid #ff7ccc;'>
            <h2 style='color:#c2185b;'>🌸 Resultado de la Clasificación</h2>
            <h3><b>{class_name}</b></h3>
            <p style='font-size:18px;'>Nivel de confianza: <b>{confidence:.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # TABLA DE PROBABILIDADES
    # -----------------------------
    prob_table = {"Flor": class_names, "Probabilidad": prediction.flatten().round(3)}
    st.table(prob_table)



# -----------------------------
# CARRUSEL DE IMÁGENES (manual)
# -----------------------------
st.subheader("🌺 Ejemplo")
flower_urls = ["https://media.admagazine.com/photos/61eb22cb9b19d943aa117b30/master/w_1600%2Cc_limit/Girasol.jpg","https://images.pexels.com/photos/64221/flower-sunflower-karnataka-india-64221.jpeg?cs=srgb&dl=pexels-pixabay-64221.jpg&fm=jpg","https://images.pexels.com/photos/65619/roses-pink-family-rose-family-65619.jpeg?cs=srgb&dl=pexels-pixabay-65619.jpg&fm=jpg"]

idx = st.slider("Mover carrusel", 0, len(flower_urls)-1, 0)
st.image(flower_urls[idx], width=350)

