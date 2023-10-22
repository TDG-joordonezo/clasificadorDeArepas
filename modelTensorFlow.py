import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 

print("Versión de TensorFlow:", tf.__version__)

def to_grayscale_from_rgb(image):
  image = tf.convert_to_tensor(image)
  #image = tf.image.rgb_to_grayscale(image)
  
  return image

#imprimir primera imagen
image = tf.io.read_file('./arepasDatasetSinFondo/defectuosas/IMG_20230820_171451984.png')
image = tf.image.decode_jpeg(image, channels=3)  # Asegurarse de que la imagen tenga 3 canales (RGB)

# Convertir a escala de grises
gray_image = to_grayscale_from_rgb(image)

# Comprobar la forma de la imagen en escala de grises
print(gray_image.shape)

# Mostrar la imagen en escala de grises
plt.imshow(gray_image, cmap='gray')

plt.show()

TAM_X_ORIG=3000
TAM_Y_ORIG=4000
X_TAM=int(TAM_X_ORIG * .08)
Y_TAM=int(TAM_Y_ORIG * .08)

datos = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. /255,
    validation_split=0.2,
    preprocessing_function=to_grayscale_from_rgb
)
path_data="./arepasDatasetSinFondo"
data_entrenamiento = datos.flow_from_directory(path_data,
                                               target_size=(X_TAM, Y_TAM),
                                               color_mode="grayscale",
                                               batch_size=32,
                                               shuffle=True,
                                               class_mode='categorical',
                                               subset="training"
                                               )

data_pruebas = datos.flow_from_directory(path_data,
                                               target_size=(X_TAM, Y_TAM),
                                               color_mode="grayscale",
                                               batch_size=32,
                                               shuffle=True,
                                               class_mode='categorical',
                                               subset="validation"
                                               )

#ver imagen de prueba en plt
plt.figure(figsize=(10,10))
for i, (imagenes, etiquetas) in enumerate(data_entrenamiento):
    print("Shape: ",imagenes[0].shape)
    if i <= 25:
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagenes[i], cmap='gray')
        break
    else:
       break  
plt.show()
    
print(data_entrenamiento.class_indices)

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(X_TAM,Y_TAM,1), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

tensorboardCNN = keras.callbacks.TensorBoard(log_dir='logs/cnn')
EPOCAS = 200
entrenamiento = modelo.fit(
    data_entrenamiento,
    batch_size=32,
    steps_per_epoch=int(np.ceil(data_entrenamiento.n / float(32))),
    epochs=EPOCAS,
    validation_data=data_pruebas,
    validation_steps=int(np.ceil(data_pruebas.n / float(32))),
    callbacks=[tensorboardCNN]
)