#Librerias 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import pathlib
from keras.applications.inception_v3 import InceptionV3
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing import image_dataset_from_directory
from collections import Counter
import random
#----------------------------------------------------------------------------------------
#-------------------------------------Preámbulo-------------------------------------
#Parámetros
img_height,img_width = 299,299 # tamaño de redimension de lasi magenes
rate = 0.001 # taza de aprendizaje para el entrenamiento
batch_size = 8 # tamaño de lote
epochs = 500 # epocas para el entrenamiento
ejecucion=10
seed=[11,123,5,901,49,231,501,7,4141,33]
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
    #rescale=1./255,
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
    rotation_range=55, 
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.5, 1.5),
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    
)

## Reescalado para validación y prueba
datagen_val_test = ImageDataGenerator(
    #rescale=1./255,
  preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
)

#Ruta de los datos 
data_dir ='/home/mocs/data/DataSet_Pineapple_Part1' # imagenes del conjunto
 #-------------------------------Codigo Base-------------------------------------------------------- 
Tiempo_ejec=time.time()
for i in range(ejecucion): #Inician las ejecuciones
    print(f"Ejecucion numero {i+1}:") 
    print(f"Semilla: {seed[i]}:") 
     # Establecer semillas para asegurar reproducibilidad
    np.random.seed(seed[i])
    tf.random.set_seed(seed[i])
    random.seed(seed[i])
  # Carga el conjunto de datos de la ruta data dir 
    dataset_total = image_dataset_from_directory(
      data_dir,
      image_size=(img_height, img_width),
      batch_size=batch_size, # Se dividen en lote 
      label_mode='binary', # las etiquetas con 0 y 1
      shuffle=True, #No se mezclan los datos 
      seed=seed[i]
    )

    # Obtener la lista de nombres de clase
    class_names = dataset_total.class_names
    print("Nombres de las clases:", class_names)
    print(f"La clase 0 corresponde a: {class_names[0]}")
    print(f"La clase 1 corresponde a: {class_names[1]}")



    #Crear archivos para almacenar informacion
    #ruta1 = f'/home/mocs/src/Red_InceptionV3_historial_{rate}_{batch_size}_{epochs}_{i+1}_{seed[i]}.txt'
    #ruta2= f'/home/mocs/src/Red_InceptionV3_resumen_{rate}_{batch_size}_{epochs}_{i+1}_{seed[i]}.txt'
    #directorio = os.path.dirname(ruta1)
    #if not os.path.exists(directorio):
    #   os.makedirs(directorio)
    #directorio = os.path.dirname(ruta2)
    #if not os.path.exists(directorio):
    #   os.makedirs(directorio)

    #Arreglos para almacenar datos 
    min_train_accuracy=[]
    max_train_accuracy=[]
    min_val_accuracy=[]
    max_val_accuracy=[]
    train_loss_final=[]
    train_accuracy_final=[]
    val_loss_final=[]
    val_accuracy_final=[]  
    resultados1 = []
    resultados2=[]
    resultados3=[]
    resultados4=[]
    resultados5=[]
    resultados6=[]
    matrices_confusion = []
    etiquetas_verdaderas = [] 

    # Convertir el conjunto de datos en listas numpy
    images = [] # lista de imagenes
    labels = [] # lista de etiquetas 


    for image_batch, label_batch in dataset_total:
        images.extend(image_batch)
        labels.extend(label_batch.numpy().flatten())

    # Convertir listas a numpy arrays
    images = np.array(images)
    labels = np.array(labels)


    # Dividir el conjunto de datos en entrenamiento y prueba (90-10) de manera estratificada
    images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.1, stratify=labels, random_state=seed[i]
    )

    # Crear conjunto de prueba usando `datagen_val_test
    test_data_generator = datagen_val_test.flow(images_test, labels_test, batch_size=batch_size, shuffle=True, seed=seed[i])
    #Etiquetas del conjunto de prueba
    for j in range(len(test_data_generator)):
       images, etiquetas = test_data_generator[j]  
       etiquetas_verdaderas.extend(etiquetas) 
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
    model.compile(optimizer=Adam(learning_rate=rate), #se emplea el optimizador Adam con tasa de aprendizaje 0.001
                      loss=BinaryCrossentropy(from_logits=False),   # función de pérdida
                      metrics=['accuracy']# metrica de precisión
    )
    time_initial= time.time()
    #Validación cruzada
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed[i])
    predicciones_acumuladas = np.zeros((len(labels_test), 1))
    previous_train_index = None
    previous_val_index = None
    #with open(ruta1, 'w') as f:
    for fold, (train_index, val_index) in enumerate(kf.split(images_train, labels_train)):
          print(f'Inicia Fold {fold + 1}, ejecucion {i+1}:\n')
          # Reiniciar los pesos del modelo a los iniciales
          model.set_weights(initial_weights)
          # Dividir el conjunto de datos train total para cada fold en train y val
          train_images_fold, val_images_fold = np.array(images_train)[train_index], np.array(images_train)[val_index]
          train_labels_fold, val_labels_fold = np.array(labels_train)[train_index], np.array(labels_train)[val_index]

          # Verificación de disjunción entre conjuntos de entrenamiento y validación
          assert len(set(train_index) & set(val_index)) == 0, "Los conjuntos de entrenamiento y validación no son disjuntos"

          # Verificación de que el conjunto de entrenamiento del fold i es diferente al fold i+1
          if previous_train_index is not None:
             assert len(set(train_index) & set(previous_train_index)) == 0, "Los conjuntos de entrenamiento del fold actual y el anterior no son disjuntos"
            
          # Verificación de que el conjunto de validación del fold i es diferente al fold i+1
             assert len(set(val_index) & set(previous_val_index)) == 0, "Los conjuntos de validación del fold actual y el anterior no son disjuntos"

          previous_train_index = train_index  # Actualizar el índice del conjunto de entrenamiento anterior
          previous_val_index = val_index      # Actualizar el índice del conjunto de validación anterior

          
          # Aplica el aumento de datos únicamente al conjunto de entrenamiento
          train_fold_generator=datagen.flow(train_images_fold,train_labels_fold, batch_size=batch_size, shuffle=True, seed=seed[i]) 
     
          ## Convertir el conjunto de validación en tensor y aplicar `datagen_val_test`
          val_data_fold = datagen_val_test.flow(val_images_fold, val_labels_fold, batch_size=batch_size, shuffle=True, seed=seed[i])
        
     
          #Estructura del modelo
          #model.compile(optimizer=Adam(learning_rate=rate), #se emplea el optimizador Adam con tasa de aprendizaje 0.001
          #            loss=BinaryCrossentropy(from_logits=False),   # función de pérdida
          #            metrics=['accuracy']# metrica de precisión
          #)
     
           # Entrenar el modelo
          history = model.fit(
            train_fold_generator,
            epochs=epochs,  # Número de épocas de entrenamiento
            validation_data=val_data_fold, shuffle=True
          )