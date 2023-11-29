from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import seaborn as sns

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

modelName= 'kaggle-mobilenet-v2-T224Negro+Transferlearning+D250+D100+E50+S1+DATE=1701220293.2954206'
url = 'https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/tf2-preview-feature-vector/versions/4'

def custom_hub_layer(url, input_shape=(224,224,3), **kwargs):
    return hub.KerasLayer(url, input_shape=input_shape, **kwargs)

custom_objects = {'KerasLayer': custom_hub_layer(url)}
loaded_model = keras.models.load_model('./modelos/' + modelName + '/keras/modelo.h5', custom_objects=custom_objects)

#print("Versión de TensorFlow.js:", tfjs.__version__)

#tfjs.converters.save_keras_model(loaded_model, ('./modelos/'+modelName+'/tfjs1'))

test_score = loaded_model.evaluate(data_test)
print("Test loss:", test_score[0])
print("Test accuracy:", test_score[1])
# Inicializar listas para acumular predicciones y etiquetas
all_predictions = []
all_labels = []

# Obtener predicciones y etiquetas por lotes
for _ in range(len(data_test)):
    images, labels = next(data_test)
    predictions = loaded_model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    all_predictions.extend(predicted_labels)
    all_labels.extend(np.argmax(labels, axis=1))

# Convertir listas a arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calcular la matriz de confusión
conf_mat = confusion_matrix(all_labels, all_predictions)

# Visualizar la matriz de confusión en un plot
def plot_confusion_matrix(conf_mat, class_names):
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.title('Matriz de Confusión')
    plt.show()

class_names = ["Defectuosa", "Perfecta"]
plot_confusion_matrix(conf_mat, class_names)

f1 = f1_score(all_labels, all_predictions, average='weighted')
print("F1-Score:", f1)

class_report = classification_report(all_labels, all_predictions, target_names=class_names)
print("Resumen de clasificación:")
print(class_report)

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


