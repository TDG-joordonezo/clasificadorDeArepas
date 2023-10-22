import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Versión de TensorFlow:", tf.__version__)

def to_grayscale_from_rgb(image):
  image = tf.convert_to_tensor(image)  
  return image

#imprimir primera imagen
image = tf.io.read_file('./arepasDatasetSinFondo/defectuosas/IMG_20230820_171451984.png')
image = tf.image.decode_jpeg(image, channels=3)
gray_image = to_grayscale_from_rgb(image)
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')
plt.show()

TAM_X_ORIG=3000
TAM_Y_ORIG=4000
X_TAM=int(TAM_X_ORIG * .08)
Y_TAM=int(TAM_Y_ORIG * .08)
BATCH_SIZE=32

datos = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. /255,
    validation_split=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    zoom_range=[.5,1.05],
    shear_range=5,
    preprocessing_function=to_grayscale_from_rgb
)

path_data="./arepasDatasetSinFondo"
data_entrenamiento = datos.flow_from_directory(path_data,
                                               target_size=(X_TAM, Y_TAM),
                                               color_mode="grayscale",
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               class_mode='categorical',
                                               subset="training"
                                               )

data_validacion = datos.flow_from_directory(path_data,
                                               target_size=(X_TAM, Y_TAM),
                                               color_mode="grayscale",
                                               batch_size=BATCH_SIZE,
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

modelo = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), input_shape=(X_TAM,Y_TAM,1), activation="relu"),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(250, activation="relu"),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

tensorboardCNN = keras.callbacks.TensorBoard(log_dir='logs/cnnNoDout250')
EPOCAS = 50
entrenamiento = modelo.fit(
    data_entrenamiento,
    batch_size=BATCH_SIZE,
    steps_per_epoch=int(np.ceil(data_entrenamiento.n / float(BATCH_SIZE))),
    epochs=EPOCAS,
    validation_data=data_validacion,
    validation_steps=int(np.ceil(data_validacion.n / float(BATCH_SIZE))),
    callbacks=[tensorboardCNN]
)