import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
import tensorflow_hub as hub
import time

print("Versión de TensorFlow:", tf.__version__)


def to_grayscale_from_rgb(image):
    image = tf.convert_to_tensor(image)
    return image


# imprimir primera imagen
image = tf.io.read_file(
    './DatasetNuevoCuadrado336pxNegro\entrenamiento\defectuosas\IMG_20231104_002007554.jpg')
image = tf.image.decode_jpeg(image, channels=3)
gray_image = to_grayscale_from_rgb(image)
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')
plt.show()

TAM_X_ORIG = 336
TAM_Y_ORIG = 336
X_TAM = int(TAM_X_ORIG * 0.667)
Y_TAM = int(TAM_Y_ORIG * 0.667)
BATCH_SIZE = 32
tf.random.set_seed(1)

train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    width_shift_range=0.01,
    height_shift_range=0.03,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=.01,
    preprocessing_function=to_grayscale_from_rgb
)

validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)

path_data = "./DatasetNuevoCuadrado336pxNegro"
path_entrenamiento = path_data + "/entrenamiento"
path_pruebas = path_data + "/pruebas"
data_entrenamiento = train_data_generator.flow_from_directory(path_entrenamiento,
                                                              target_size=(
                                                                  X_TAM, Y_TAM),
                                                              #color_mode="grayscale",
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              class_mode='categorical',
                                                              subset="training"
                                                              )

data_validacion = validation_data_generator.flow_from_directory(path_entrenamiento,
                                                                target_size=(
                                                                    X_TAM, Y_TAM),
                                                                #color_mode="grayscale",
                                                                batch_size=BATCH_SIZE,
                                                                shuffle=True,
                                                                class_mode='categorical',
                                                                subset="validation"
                                                                )

data_test = validation_data_generator.flow_from_directory(path_pruebas,
                                                          target_size=(
                                                              X_TAM, Y_TAM),
                                                          #color_mode="grayscale",
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=False,
                                                          class_mode='categorical',
                                                          )

# ver imagen de prueba en plt
plt.figure(figsize=(24, 18))
count = 0
for imagenes, etiquetas in data_entrenamiento:
    print("Shape: ", imagenes[0].shape)
    for i in range(len(imagenes)):
        plt.subplot(5, 5, count + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagenes[i], cmap='gray')
        count += 1
        if count >= 25:
            break
    if count >= 25:
        break

plt.show()

print(data_entrenamiento.class_indices)

url = 'https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/tf2-preview-feature-vector/versions/4'

modelo = keras.Sequential([
    hub.KerasLayer(url, input_shape=(224,224,3), trainable=False),
    # keras.layers.Conv2D(32, (3, 3), input_shape=(X_TAM, Y_TAM, 3), activation="relu"),
    # keras.layers.MaxPooling2D(2, 2),

    # keras.layers.Conv2D(64, (3, 3), activation="relu"),
    # keras.layers.MaxPooling2D(2, 2),

    # keras.layers.Conv2D(128, (3, 3), activation="relu"),
    # keras.layers.MaxPooling2D(2, 2),

    # keras.layers.Flatten(),
    keras.layers.Dense(250, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print(modelo.summary())
EPOCAS = 50
modelName = 'kaggle-mobilenet-v2-T224Negro+Transferlearning+D250+D100+E'+str(EPOCAS)+'+S1+DATE='+str(time.time())
bestModel = 'T224Negro+Transferlearning+D250+E50+S1'
tensorboardCNN = keras.callbacks.TensorBoard(log_dir=('logs/'+modelName))
model_checkpoint = keras.callbacks.ModelCheckpoint(f'./modelos/{bestModel}/keras/modelo.h5', 
                                   save_best_only=True,
                                   monitor='val_accuracy',
                                   mode='max')
entrenamiento = modelo.fit(
    data_entrenamiento,
    batch_size=BATCH_SIZE,
    steps_per_epoch=int(np.ceil(data_entrenamiento.n / float(BATCH_SIZE))),
    epochs=EPOCAS,
    validation_data=data_validacion,
    validation_steps=int(np.ceil(data_validacion.n / float(BATCH_SIZE))),
    callbacks=[tensorboardCNN, model_checkpoint]
)
print("Guardando model .h5")
modelo.save(('./modelos/'+modelName+'/keras/modelo.h5'))
score = modelo.evaluate(data_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# guardar modelo en tensorflowjs
print("Guardando model tfjs")
tfjs.converters.save_keras_model(modelo, ('./modelos/'+modelName+'/tfjs'))
print("Finalizó todo correctamente!!!")
