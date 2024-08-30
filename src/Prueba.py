#Librerias 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import pathlib
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
import time
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing import image_dataset_from_directory
from collections import Counter

#Preámbulo
#Parámetros
img_height,img_width = 299,299 # tamaño de redimension de lasi magenes
rate = 0.001 # taza de aprendizaje para el entrenamiento
batch_size = 32 # tamaño de lote
epochs = 200 # epocas para el entrenamiento

#Funcion del modelo base 
def create_modelo_base():
     modelo_base=InceptionV3(include_top=False,
     weights="imagenet",
     input_shape=(img_height, img_width, 3),
     pooling='avg',
     classes=2)
     return modelo_base
#Creacion del modelo de CNN para entrenamiento
def create_model():
    modelo_base=create_modelo_base()
    modelo_base.trainable = False
    model_Inceptionv3 = Sequential([
      modelo_base,
      Flatten(),
      Dense(1, activation='sigmoid')
    ])
    return model_Inceptionv3

#Funcion para generar el aumento de datos 
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=55, 
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    #preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
)

## Reescalado y preprocesamiento para validación y prueba
datagen_val_test = ImageDataGenerator(
    rescale=1./255,
  #preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
)

#Crear archivos para almacenar informacion
ruta1 = f'/home/mocs/src/InceptionV3_history_{rate}_{batch_size}_{epochs}_Final.txt'
ruta2= f'/home/mocs/src/InceptionV3_resumen_{rate}_{batch_size}_{epochs}_Final.txt'
directorio = os.path.dirname(ruta1)
if not os.path.exists(directorio):
    os.makedirs(directorio)
directorio = os.path.dirname(ruta2)
if not os.path.exists(directorio):
    os.makedirs(directorio)

#Arreglos para almacenar datos 
min_train_accuracy=[]
max_train_accuracy=[]
min_val_accuracy=[]
max_val_accuracy=[]
modelos=[]   
results = []
resultados = []
matrices_confusion = []
etiquetas_verdaderas = [] 
#*-------------------------------------------------------------------------------------------*
#Código base 
#Ruta de los datos
data_dir ='/home/mocs/data/DataSet_Pineapple_Part1' # imagenes del conjunto
# Carga el conjunto de datos de la ruta data dir 
dataset_total = image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size, # Se dividen en lote 
    label_mode='binary', # las etiquetas con 0 y 1
    shuffle=False #No se mezclan los datos 
)

# Obtener la lista de nombres de clase
class_names = dataset_total.class_names
print("Nombres de las clases:", class_names)
print(f"La clase 0 corresponde a: {class_names[0]}")
print(f"La clase 1 corresponde a: {class_names[1]}")

# Convertir el conjunto de datos en listas numpy
images = [] # lista de imagenes
labels = [] # lista de etiquetas 


for image_batch, label_batch in dataset_total:
    images.extend(image_batch)
    labels.extend(label_batch.numpy().flatten())

# Convertir listas a numpy arrays
images = np.array(images)
labels = np.array(labels)
# Verificar la longitud de las listas
print("Numero de imagenes:", len(images))
print("Numero de etiquetas:", len(labels))
print(labels)
# Contar las etiquetas
label_counts = Counter(labels)
print(f"Distribucion de clases: {label_counts}")

# Dividir el conjunto de datos en entrenamiento y prueba (90-10) de manera estratificada
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.1, stratify=labels, random_state=42  # Ajusta random_state si deseas
)
#Convertir el conjunto de prueba en tensor
test_data = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
test_data = test_data.batch(batch_size)#Mezclar los datos en lotes
# Crear conjuntos de datos de validación y prueba usando `datagen_val_test`
#test_data = datagen_val_test.flow(images_test, labels_test, batch_size=batch_size, shuffle=False)

# Verificar la distribución de clases en los conjuntos de entrenamiento y prueba
print(f"Conjunto de entrenamiento : {len(labels_train)}")
print(f"Prueba: {len(labels_test)}")
print("Clases en el conjunto de entrenamiento:")
print(Counter(labels_train))
print("Clases en el conjunto  de prueba:")
print(Counter(labels_test))

#------------------------------------------VALIDACIÓN CRUZADA------------------------------------
#Crear el modelo y almacenar los pesos iniciales
model = create_model()
initial_weights = model.get_weights()
time_initial= time.time()
#Validación cruzada
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
with open(ruta1, 'w') as f:
  for fold, (train_index, val_index) in enumerate(kf.split(images_train, labels_train)):
     print(f'Inicia Fold {fold + 1}:\n')
     # Reiniciar los pesos del modelo a los iniciales
     model.set_weights(initial_weights)
     # Dividir el conjunto de datos train total para cada fold en train y val
     train_images_fold, val_images_fold = np.array(images_train)[train_index], np.array(images_train)[val_index]
     train_labels_fold, val_labels_fold = np.array(labels_train)[train_index], np.array(labels_train)[val_index]
     print("Distribucion de clases en el conjunto de entrenamiento:")
     print(Counter(train_labels_fold))
    
     print("\nDistribucion de clases en el conjunto de validacion:")
     print(Counter(val_labels_fold))
      # Aplica el aumento de datos únicamente al conjunto de entrenamiento
     train_fold_generator=datagen.flow(train_images_fold,train_labels_fold, batch_size=batch_size, shuffle=True) 
     # Convertir el conjunto de validacion en tensor en tensores
     #val_data_fold = tf.data.Dataset.from_tensor_slices((val_images_fold, val_labels_fold)).batch(batch_size)
     #Opció 2
     #val_data_fold = tf.data.Dataset.from_tensor_slices((val_images_fold, val_labels_fold))
     #train_data_fold = train_data_fold.shuffle(buffer_size=len(train_images_fold)).batch(batch_size)
     ## Convertir el conjunto de validación en tensor y aplicar `datagen_val_test`
     val_data_fold = datagen_val_test.flow(val_images_fold, val_labels_fold, batch_size=batch_size, shuffle=False)
        
     
     #Estructura del modelo
     model.compile(optimizer=Adam(learning_rate=rate), #se emplea el optimizador Adam con tasa de aprendizaje 0.001
                      loss=BinaryCrossentropy(from_logits=False),   # función de pérdida
                      metrics=['accuracy']# metrica de precisión
     )
     
     # Entrenar el modelo
     history = model.fit(
        train_fold_generator,
        epochs=epochs,  # Número de épocas de entrenamiento
        validation_data=val_data_fold, shuffle=True
     )
     
     modelos.append(model)
     
     f.write(f'Fold {fold + 1}:\n')
     for key in history.history:
       f.write(f'{key}: {history.history[key]}\n')
     f.write('\n')
 

#--------------------------------------------------------------------------------
     # Obtener el menor y el mejor accuracy en el conjunto de entrenamiento y validacion
     min_train_accuracy_fold = min(history.history['accuracy'])
     max_train_accuracy_fold = max(history.history['accuracy'])
     min_val_accuracy_fold = min(history.history['val_accuracy'])
     max_val_accuracy_fold= max(history.history['val_accuracy'])
     min_train_accuracy.append(min_train_accuracy_fold)
     max_train_accuracy.append(max_train_accuracy_fold)
     min_val_accuracy.append(min_val_accuracy_fold)
     max_val_accuracy.append(max_val_accuracy_fold) 
 

min_train_accuracy = np.array(min_train_accuracy)
max_train_accuracy= np.array(max_train_accuracy)
min_val_acuracy = np.array(min_val_accuracy)
max_val_accuracy = np.array(max_val_accuracy)
mean_train_min_accuracy=np.mean(min_train_accuracy)
mean_train_max_accuracy=np.mean(max_train_accuracy)
mean_val_min_accuracy=np.mean(min_val_accuracy)
mean_val_max_accuracy=np.mean(max_val_accuracy)

time_end = time.time()
Time = time_end-time_initial 
Tiempo=Time/3600

#--------------------EVALUACION DEL MODELO-----------------------------
for imagenes, etiquetas in test_data:
    etiquetas_verdaderas.extend(etiquetas.numpy())

for i, model in enumerate(modelos):
    test_loss, test_acc = model.evaluate(test_data)
    predictions = model.predict(test_data)
    auc_roc = roc_auc_score(etiquetas_verdaderas, predictions)
    results.append({'modelo': i+1, 'loss_test': test_loss, 'accuracy_test': test_acc, 'auc_roc': auc_roc})
    resultados.append(test_acc)

    if i == 0:
        predicciones_acumuladas = np.zeros_like(predictions)

    predicciones_acumuladas += predictions

    predicted_classes = (predictions > 0.5).astype(int)
    conf_matrix = confusion_matrix(etiquetas_verdaderas, predicted_classes)
    print(conf_matrix)
    matrices_confusion.append(conf_matrix)
test_mean=np.mean(resultados)
predicciones_acumuladas /= len(modelos)
predicted_classes_final = (predicciones_acumuladas > 0.5).astype(int)
final_accuracy = np.mean(predicted_classes_final == np.array(etiquetas_verdaderas))
print(f"Promedio de la precision en el conjunto de prueba:{test_mean}")
print(f"Precision promedio final en el conjunto de prueba: {final_accuracy}")
print("Tiempo de entrenamiento:", Tiempo)
conf_matrix_final = confusion_matrix(etiquetas_verdaderas, predicted_classes_final)
print(conf_matrix_final)

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
    archivo.write(f"Promedio de la precision en el conjunto de prueba: {test_mean}\n")
    archivo.write(f"Matriz de confusion final:\n{conf_matrix_final}\n")
    archivo.write(f"Tiempo de entrenamiento:{Tiempo}\n")
   
