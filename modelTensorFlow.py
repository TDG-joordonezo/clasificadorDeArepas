import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflowjs as tfjs

print("Versión de TensorFlow:", tf.__version__)


def to_grayscale_from_rgb(image):
    image = tf.convert_to_tensor(image)
    return image


# imprimir primera imagen
image = tf.io.read_file(
    './arepasDatasetMejoradas\entrenamiento\defectuosas\IMG_20231104_002007554.jpg')
image = tf.image.decode_jpeg(image, channels=3)
gray_image = to_grayscale_from_rgb(image)
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')
plt.show()

TAM_X_ORIG = 240
TAM_Y_ORIG = 320
X_TAM = int(TAM_X_ORIG * 1)
Y_TAM = int(TAM_Y_ORIG * 1)
BATCH_SIZE = 32
tf.random.set_seed(1)

train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1,
    width_shift_range=0.01,
    height_shift_range=0.03,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=.01,
    preprocessing_function=to_grayscale_from_rgb
)

validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1,
)

path_data = "./arepasDatasetMejoradas"
path_entrenamiento = path_data + "/entrenamiento"
path_pruebas = path_data + "/pruebas"
data_entrenamiento = train_data_generator.flow_from_directory(path_entrenamiento,
                                                              target_size=(
                                                                  X_TAM, Y_TAM),
                                                              color_mode="grayscale",
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              class_mode='categorical',
                                                              subset="training"
                                                              )

data_validacion = validation_data_generator.flow_from_directory(path_entrenamiento,
                                                                target_size=(
                                                                    X_TAM, Y_TAM),
                                                                color_mode="grayscale",
                                                                batch_size=BATCH_SIZE,
                                                                shuffle=True,
                                                                class_mode='categorical',
                                                                subset="validation"
                                                                )

data_test = validation_data_generator.flow_from_directory(path_pruebas,
                                                          target_size=(
                                                              X_TAM, Y_TAM),
                                                          color_mode="grayscale",
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

modelo = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(X_TAM, Y_TAM, 1), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(250, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
EPOCAS = 100
modelName = '3Xfilter+D250+E'+str(EPOCAS)+'+S1'
tensorboardCNN = keras.callbacks.TensorBoard(log_dir=('logs/'+modelName))
entrenamiento = modelo.fit(
    data_entrenamiento,
    batch_size=BATCH_SIZE,
    steps_per_epoch=int(np.ceil(data_entrenamiento.n / float(BATCH_SIZE))),
    epochs=EPOCAS,
    validation_data=data_validacion,
    validation_steps=int(np.ceil(data_validacion.n / float(BATCH_SIZE))),
    callbacks=[tensorboardCNN]
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
