import tensorflow as tf
from tensorflow import keras
import PIL
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ruta_dataset = '/home/bloom/Documents/UNALM/2025-2/CD1'


# Ruta al train_test_split
dirtreno = '/train'
dirprueba =  '/test'

datagen = ImageDataGenerator(rescale = 1./255,
                             validation_split = 0.2,
                             rotation_range = 20,
                             width_shift_range = 0.10,
                             height_shift_range= 0.10,
                             zoom_range = 0.15,
                             horizontal_flip = True,
                             shear_range = 0.10,
                             # Suavizar contraste
                             brightness_range = [0.9, 1.1]
                             )

# Generador de entrenamiento con augmentation
trenogen = datagen.flow_from_directory(
        dirtreno,
        target_size = (150,150), # Redimensionamiento
        batch_size = 32,
        class_mode = "categorical",
        shuffle = True,
        subset = 'training'
)

# Generador de validación sin augmentation (solo reescalado)
validatagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
)

validaciongen = validatagen.flow_from_directory(
        dirtreno,
        target_size = (150,150),
        batch_size = 32,
        class_mode = "categorical",
        shuffle = False,
        subset = 'validation'
)

print("Clases detectadas: ", trenogen.class_indices)

