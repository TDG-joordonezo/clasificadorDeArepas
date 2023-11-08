from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

TAM_X_ORIG = 240
TAM_Y_ORIG = 320
X_TAM = int(TAM_X_ORIG * 1)
Y_TAM = int(TAM_Y_ORIG * 1)
BATCH_SIZE = 32


validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1,
)
path_data = "./arepasDatasetMejoradas"
path_pruebas = path_data + "/pruebas"

data_test = validation_data_generator.flow_from_directory(path_pruebas,
                                                          target_size=(
                                                              X_TAM, Y_TAM),
                                                          color_mode="grayscale",
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=False,
                                                          class_mode='categorical',
                                                          )

modelName= 'Cruzado3-dataAument-split0-1+3Xfilter+D250+E100+S1'
loaded_model = keras.models.load_model('./modelos/' + modelName + '/keras/modelo.h5')

test_score = loaded_model.evaluate(data_test)
print("Test loss:", test_score[0])
print("Test accuracy:", test_score[1])

predictions = loaded_model.predict(data_test)
images, labels = next(data_test)
predicted_labels = np.argmax(predictions, axis=1)
class_names = ["Perfecta", "Defectuosa"]

def plot_images(images, labels, predicted_labels, class_names):
    plt.figure(figsize=(12, 12))
    for i in range(min(len(images), 9)):  # Mostrar hasta 9 imágenes
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i][0], cmap='gray')
        actual_label = np.argmax(labels[i])
        predicted_label = predicted_labels[i]
        plt.xlabel(f"Real: {class_names[actual_label]}\nPredicción: {class_names[predicted_label]}")
    plt.show()

plot_images(images, labels, predicted_labels, class_names)
