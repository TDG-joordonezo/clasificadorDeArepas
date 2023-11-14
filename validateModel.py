from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

TAM_X_ORIG = 336
TAM_Y_ORIG = 336
X_TAM = int(TAM_X_ORIG * 0.667)
Y_TAM = int(TAM_Y_ORIG * 0.667)
BATCH_SIZE = 32


validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)
path_data = "./DatasetNuevoCuadrado336pxNegro"
path_pruebas = path_data + "/pruebas"

data_test = validation_data_generator.flow_from_directory(path_pruebas,
                                                          target_size=(
                                                              X_TAM, Y_TAM),
                                                          #color_mode="grayscale",
                                                          batch_size=BATCH_SIZE,
                                                          shuffle=False,
                                                          class_mode='categorical',
                                                          )

modelName= 'T224Negro+Transferlearning+D250+E50+S1'
url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
def custom_hub_layer(url, input_shape=(224,224,3), **kwargs):
    return hub.KerasLayer(url, input_shape=input_shape, **kwargs)
custom_objects = {'KerasLayer': custom_hub_layer(url)}
loaded_model = keras.models.load_model('./modelos/' + modelName + '/keras/modelo.h5', custom_objects=custom_objects)

test_score = loaded_model.evaluate(data_test)
print("Test loss:", test_score[0])
print("Test accuracy:", test_score[1])

predictions = loaded_model.predict(data_test)
images, labels = next(data_test)
predicted_labels = np.argmax(predictions, axis=1)
class_names = ["Defectuosa","Perfecta"]

def plot_images(images, labels, predicted_labels, class_names, items_per_page=9, margin=0.1):
    num_items = len(images)
    num_pages = -(-num_items // items_per_page)

    for page in range(num_pages):
        plt.figure(figsize=(12, 12))

        for i in range(items_per_page):
            index = page * items_per_page + i
            if index < num_items:
                plt.subplot(3, 3, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(images[index], cmap='gray')

                actual_label = np.argmax(labels[index])
                predicted_label = predicted_labels[index]

                color = 'green' if actual_label == predicted_label else 'red'

                plt.xlabel(
                    f"Real: {class_names[actual_label]}\nPredicción: {class_names[predicted_label]}",
                    color=color,
                    bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=' + str(margin))
                )

        plt.tight_layout()
        plt.show()

items_per_page = 9
plot_images(images, labels, predicted_labels, class_names, items_per_page=items_per_page)


