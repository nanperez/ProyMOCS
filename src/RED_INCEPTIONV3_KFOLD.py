#Librerias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import pathlib
from keras.applications.inception_v3 import InceptionV3
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import KFold

#Ruta de los datos
data_dir ='/home/mocs/data/DataSet_Blue_Pineapple_Part1' # imagenes del conjunto

img_height = 299
img_width = 299
batch_size = 32
epochs = 50
rate = 0.01

#Generar aumento de datos 
datagen = ImageDataGenerator(
    rescale=1./255, # reescalar
    rotation_range=40, #rotación 
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)


#Cargar el conjunto de datos desde la carpeta
dataset = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)
# Obtener la lista de nombres de clase
class_names = dataset.class_indices

# Imprimir los nombres de las clases
print("Nombres de las clases:", class_names)

# Convertir el conjunto de datos en listas numpy
images = []
labels = []

for _ in range(len(dataset)):
    image_batch, label_batch = next(dataset)
    images.extend(image_batch)
    labels.extend(label_batch)
    
    
#Divide los datos en entrenamiento Y prueba
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.1, random_state=42)
# Convierte las listas test nuevamente en tensores
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
#Mezclar los datos  en lotes
test_data = test_data.batch(batch_size)

# Imprime cuántas imágenes pertenecen a cada conjunto
print(f"Entrenamiento: {len(train_labels)}")
#print(f"Validación: {len(val_labels)}")
print(f"Prueba: {len(test_labels)}")

#Modelo Inception_v3
#Configuración del modelo
#Entrenamiento
model=InceptionV3(include_top=False,
     weights="imagenet",
     input_shape=(img_height, img_width, 3),
     pooling='avg',
     classes=2)
# Congelar todas las capas del modelo base vgg16
model.trainable = False

#Estructura del modelo
model_Inceptionv3 = Sequential([
    model,
    Flatten(),
    Dense(1, activation='sigmoid')
])



model_Inceptionv3.compile(optimizer=Adagrad(learning_rate=rate), #se emplea el optimizador Adam con tasa de aprendizaje 0.001
                      loss=BinaryCrossentropy(from_logits=False),   # función de pérdida
                      metrics=['accuracy']# metrica de precisión

)

ruta1 = '/home/mocs/src/Inception_Entrenamiento_03_history_0.01_32_b.txt'
ruta2= '/home/mocs/src/Inception_Entrenamiento_03_RESUMEN_0.01_32_b.txt'
directorio = os.path.dirname(ruta1)
if not os.path.exists(directorio):
    os.makedirs(directorio)
directorio = os.path.dirname(ruta2)
if not os.path.exists(directorio):
    os.makedirs(directorio)
    
#Incorporación de la validación cruzada
k = 5
kf = KFold(n_splits=k)
inicio= time.time()
min_train_accuracy=[]
max_train_accuracy=[]
min_val_accuracy=[]
max_val_accuracy=[]

# Entrenar y validar el modelo utilizando validación cruzada
train_images = np.array(train_images)
train_labels = np.array(train_labels)
inicio= time.time()
with open(ruta1, 'w') as f:
  for fold, (train_index, val_index) in enumerate(kf.split(train_images)):
  #for fold, (train_index, val_index) in kf.split(train_images):  # Carga el conjunto train_ imagenes para dividirlo
    train_images_fold, val_images_fold = train_images[train_index], train_images[val_index]
    train_labels_fold, val_labels_fold = train_labels[train_index], train_labels[val_index]
    # Convierte las listas nuevamente en tensores
    train_data_fold = tf.data.Dataset.from_tensor_slices((train_images_fold, train_labels_fold))
    val_data_fold = tf.data.Dataset.from_tensor_slices((val_images_fold, val_labels_fold))

    # Mezcla los datos en lotes
    train_data_fold = train_data_fold.shuffle(buffer_size=len(train_images_fold)).batch(batch_size) #Conjunto de entrenamineto
    val_data_fold = val_data_fold.batch(batch_size) # conjunto de vaidación

    # Entrenar el modelo
    history = model_Inceptionv3.fit(
        train_data_fold,
        epochs=epochs,  # Número de épocas de entrenamiento
        validation_data=val_data_fold
    )
    f.write(f'Fold {fold + 1}:\n')
    for key in history.history:
      f.write(f'{key}: {history.history[key]}\n')
    f.write('\n')


    # Obtener el menor y el mejor accuracy en el conjunto de entrenamiento
    min_train_accuracy_k = min(history.history['accuracy'])
    max_train_accuracy_k = max(history.history['accuracy'])

    # Obtener el menor y el mejor accuracy en el conjunto de validación
    min_val_accuracy_k = min(history.history['val_accuracy'])
    max_val_accuracy_k = max(history.history['val_accuracy'])
    min_train_accuracy.append(min_train_accuracy_k)
    max_train_accuracy.append(max_train_accuracy_k)
    min_val_accuracy.append(min_val_accuracy_k)
    max_val_accuracy.append(max_val_accuracy_k)
min_train_accuracy = np.array(min_train_accuracy)
max_train_accuracy= np.array(max_train_accuracy)
min_val_acuracy = np.array(min_val_accuracy)
max_val_accuracy = np.array(max_val_accuracy)
mean_train_min_accuracy=np.mean(min_train_accuracy)
mean_train_max_accuracy=np.mean(max_train_accuracy)
mean_val_min_accuracy=np.mean(min_val_accuracy)
mean_val_max_accuracy=np.mean(max_val_accuracy)

fin= time.time()
tiempo=fin-inicio


# Evaluar el modelo con los datos de prueba

test_loss, test_accuracy = model_Inceptionv3.evaluate(test_data)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Obtener las predicciones del modelo para el conjunto de prueba
predictions=model_Inceptionv3.predict(test_data)
predicted_classes = np.around(predictions)
#print(predicted_classes)
#obtener la etiquetas verdaderas
etiquetas_verdaderas = []
for imagenes, etiquetas in test_data:

    etiquetas_verdaderas.extend(etiquetas.numpy())
# Calcular la matriz de confusión
conf_matrix = confusion_matrix(etiquetas_verdaderas, predicted_classes)
print(etiquetas_verdaderas)
print("Matriz de Confusion:")
#print(conf_matrix)

# Crear un DataFrame de la matriz de confusión
df = pd.DataFrame(conf_matrix)

# Configurar el estilo del mapa de calor
sn.set(font_scale=1)

# Crear el mapa de calor
heatmap = sn.heatmap(df, annot=True, annot_kws={"size": 20}, cmap='BuPu')
plt.savefig('/home/mocs/src/matriz_confusion_inception_03_0.01_32_b.png')
plt.show(heatmap)

# Almacenar valores del entrenamiento
with open(ruta2, 'w') as archivo:
    # Escribe lo que necesites en el archivo
    archivo.write(f"Min accuracy train:{min_train_accuracy}\n")
    archivo.write(f"Max accuracy train:{max_train_accuracy}\n")
    archivo.write(f"Min accuracy val:{min_val_acuracy}\n")
    archivo.write(f"Max accuracy val:{max_val_accuracy }\n")
    archivo.write(f"Promedio  min accuracy train:{mean_train_min_accuracy}\n")
    archivo.write(f"Promedio max accuracy train:{mean_train_max_accuracy}\n")
    archivo.write(f"Promedio min accuracy val:{mean_val_min_accuracy}\n")
    archivo.write(f"Promedio max accuracy val:{mean_val_max_accuracy }\n")
    archivo.write(f"Pérdida test:{test_loss}\n")
    archivo.write(f"Accuracy test:{test_accuracy}\n")
    archivo.write(f"Matriz de confusión test:{conf_matrix}\n")
    archivo.write(f"Tiempo de entrenamiento:{tiempo}\n")
    
#Guardar el modelo
#model_RESNET50.save('RESNET50_0.001_32_c.h5')
model_Inceptionv3.save('/home/mocs/src/Inceptionv3_03_0.01_32_b.keras')
