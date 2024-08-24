#Librerias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
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



img_height,img_width = 299,299

def create_modelo_base():
     modelo_base=InceptionV3(include_top=False,
     weights="imagenet",
     input_shape=(img_height, img_width, 3),
     pooling='avg',
     classes=2)
     modelo_base.trainable = False
     return modelo_base
#--------------------------------------------------------------------------------

def create_model():
    modelo_base=create_modelo_base()
    model_Inceptionv3 = Sequential([
     modelo_base,
     Flatten(),
     Dropout(0.5),
     Dense(128, activation='relu'),
     Dense(1, activation='sigmoid')
    ])
    return model_Inceptionv3
#--------------------------------------------------------------------------------
#Ruta de los datos
data_dir ='/home/mocs/data/DataSet_Pineapple_Part1' # imagenes del conjunto
#--------------------------------------------------------------------------------
#Parámetros
rate = 0.001
batch_size = 32
epochs = 3

#--------------------------------------------------------------------------------

#Generar aumento de datos 
datagen = ImageDataGenerator(
    rescale=1./255, # reescalar
    rotation_range=55, #rotación 
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
#--------------------------------------------------------------------------------

#Cargar el conjunto de datos desde la carpeta
dataset = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)
#--------------------------------------------------------------------------------
# Obtener la lista de nombres de clase
class_names = dataset.class_indices
# Imprimir los nombres de las clases
print("Nombres de las clases:", class_names)
#--------------------------------------------------------------------------------
# Convertir el conjunto de datos en listas numpy
images = []
labels = []

for _ in range(len(dataset)):
    image_batch, label_batch = next(dataset)
    images.extend(image_batch)
    labels.extend(label_batch)
    
#--------------------------------------------------------------------------------    
#Divide los datos en entrenamiento Y prueba /(90-10)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.1, random_state=42)
# Convierte las listas test nuevamente en tensores
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
#Mezclar los datos  en lotes
test_data = test_data.batch(batch_size)

#--------------------------------------------------------------------------------
# Imprime cuántas imágenes pertenecen a cada conjunto
print(f"Entrenamiento: {len(train_labels)}")
print(f"Prueba: {len(test_labels)}")
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
ruta1 = f'/home/mocs/src/InceptionV3_history_{rate}_{batch_size}_{epochs}_cx.txt'
ruta2= f'/home/mocs/src/InceptionV3_resumen_{rate}_{batch_size}_{epochs}_cx.txt'
#--------------------------------------------------------------------------------
directorio = os.path.dirname(ruta1)
if not os.path.exists(directorio):
    os.makedirs(directorio)
directorio = os.path.dirname(ruta2)
if not os.path.exists(directorio):
    os.makedirs(directorio)
#--------------------------------------------------------------------------------    
#Incorporación de la validación cruzada
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
min_train_accuracy=[]
max_train_accuracy=[]
min_val_accuracy=[]
max_val_accuracy=[]
modelos=[]
# Entrenar y validar el modelo utilizando validación cruzada
train_images = np.array(train_images)
train_labels = np.array(train_labels)
inicio= time.time()
with open(ruta1, 'w') as f:
  for fold, (train_index, val_index) in enumerate(kf.split(train_images)):
   
    print(f'Inicia Fold {fold + 1}:\n')
    
    #Estructura del modelo
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=rate), #se emplea el optimizador Adam con tasa de aprendizaje 0.001
                      loss=BinaryCrossentropy(from_logits=False),   # función de pérdida
                      metrics=['accuracy']# metrica de precisión
    )

    # Carga el conjunto train_ imagenes para dividirlo
    train_images_fold, val_images_fold = train_images[train_index], train_images[val_index]
    train_labels_fold, val_labels_fold = train_labels[train_index], train_labels[val_index]
    # Convierte las listas nuevamente en tensores
    train_data_fold = tf.data.Dataset.from_tensor_slices((train_images_fold, train_labels_fold))
    val_data_fold = tf.data.Dataset.from_tensor_slices((val_images_fold, val_labels_fold))

    # Mezcla los datos en lotes
    train_data_fold = train_data_fold.shuffle(buffer_size=len(train_images_fold)).batch(batch_size) #Conjunto de entrenamineto
    val_data_fold = val_data_fold.batch(batch_size) # conjunto de vaidación

    # Entrenar el modelo
    history = model.fit(
        train_data_fold,
        epochs=epochs,  # Número de épocas de entrenamiento
        validation_data=val_data_fold
    )

#-------------------------------------------------------------------------------------------
    modelos.append(model)

   
#--------------------------------------------------------------------------------
    f.write(f'Fold {fold + 1}:\n')
    for key in history.history:
      f.write(f'{key}: {history.history[key]}\n')
    f.write('\n')
 

#--------------------------------------------------------------------------------
    # Obtener el menor y el mejor accuracy en el conjunto de entrenamiento y validacion
    min_train_accuracy_k = min(history.history['accuracy'])
    max_train_accuracy_k = max(history.history['accuracy'])
    min_val_accuracy_k = min(history.history['val_accuracy'])
    max_val_accuracy_k = max(history.history['val_accuracy'])
    min_train_accuracy.append(min_train_accuracy_k)
    max_train_accuracy.append(max_train_accuracy_k)
    min_val_accuracy.append(min_val_accuracy_k)
    max_val_accuracy.append(max_val_accuracy_k)
#--------------------------------------------------------------------------------    
min_train_accuracy = np.array(min_train_accuracy)
max_train_accuracy= np.array(max_train_accuracy)
min_val_acuracy = np.array(min_val_accuracy)
max_val_accuracy = np.array(max_val_accuracy)
mean_train_min_accuracy=np.mean(min_train_accuracy)
mean_train_max_accuracy=np.mean(max_train_accuracy)
mean_val_min_accuracy=np.mean(min_val_accuracy)
mean_val_max_accuracy=np.mean(max_val_accuracy)
#--------------------------------------------------------------------------------
fin= time.time()
tiempo=fin-inicio
#-------------------------------------------------------------------------------
#Predicción del modelo y matriz de confusión
results = []
resultados=[]
matrices_confusion=[]
for i, model in enumerate(modelos):
    print(f"Evaluando modelo {i+1}:")
    
    # Obtener las predicciones del modelo en el conjunto de prueba
    test_loss, test_acc = model.evaluate(test_data)
    
    print(f"Perdida en el conjunto de prueba: {test_loss}")
    print(f"Precision en el conjunto de prueba: {test_acc}")
    results.append({'modelo': i+1, 'loss_test': test_loss, 'accuracy_test': test_acc})
    resultados.append(test_acc)


     #Acumular las predicciones para el promedio
    predictions = model.predict(test_data)
    # Convertir las predicciones en clases
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calcular la matriz de confusión por modelo
    # Obtener las etiquetas verdaderas directamente del conjunto de prueba
    etiquetas_verdaderas = []
    for imagenes, etiquetas in test_data:
        etiquetas_verdaderas.extend(etiquetas.numpy())

    # Calcular la matriz de confusión para el modelo actual
    conf_matrix = confusion_matrix(etiquetas_verdaderas, predicted_classes)
    matrices_confusion.append(conf_matrix)

# Promediar las predicciones de todos los modelos
predicciones_acumuladas = np.zeros_like(predictions)

for model in modelos:
    preds = model.predict(test_data)
    predicciones_acumuladas += preds

predicciones_acumuladas /= len(modelos)

# Convertir las predicciones promediadas a clases finales
predicted_classes_final = np.argmax(predicciones_acumuladas, axis=1)

# Calcular la precisión final en el conjunto de prueba
final_accuracy = np.mean(predicted_classes_final == etiquetas_verdaderas)
print(f"Precision promedio final en el conjunto de prueba: {final_accuracy}")

# Calcular la matriz de confusión final
conf_matrix_final = confusion_matrix(etiquetas_verdaderas, predicted_classes_final)
#--------------------------------------------------------------------------------
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
    archivo.write(f"Perdida, acurracy test por cada k-fold:{results}\n")
    archivo.write(f":Matrices de confusion por k-fold: {matrices_confusion}\n")
    archivo.write(f"Precision promedio final en el conjunto de prueba: {final_accuracy}\n")
    archivo.write(f"Matriz de confusion final:\n{conf_matrix_final}\n")
    archivo.write(f"Tiempo de entrenamiento:{tiempo}\n")
    