import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
# import pytorch_lightning as pl

scaler = StandardScaler()

def get_data_from_datasets(datos, tags):
    data = pd.read_csv(datos).fillna(0)
    labels = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta', 'Tipo Carretera'])
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data_norm.reset_index(drop=True, inplace=True)

    df = pd.concat([data_norm, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    User1_data_no_time = df_reduced[[]]
    User1_data = df_reduced[['Tiempo', 'Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    Label_New = df_reduced[['Tiempo', 'Etiqueta']]

    print("Hecho!\n")
    return User1_data, Label_New

def divide_data_arrays_into_windows(dataset, tags, moda, window_size):
    dataset_no_time = dataset[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    tags_no_time = tags[['Etiqueta']]
    num_windows = math.floor(len(dataset)/window_size)
    dataset_to_numpy = dataset_no_time.to_numpy()
    tags_to_numpy = tags_no_time.to_numpy()
    x_windows = []
    y = []
    tam_arrays = len(dataset_to_numpy)

    for i in range(0, tam_arrays - window_size, window_size):
        elem = dataset_to_numpy[i:i+window_size]
        x_windows.append(elem)

    if moda is False:
        for i in range(0, tam_arrays - window_size, window_size):
            tag = tags_to_numpy[i+window_size-1] #cogemos como etiqueta del conjunto la etiqueta del último elemento
            y.append(tag)
    else:
        y_windows = []
        for i in range(0, tam_arrays - window_size, window_size):
            tag = tags_to_numpy[i:i+window_size] #cogemos como etiqueta del conjunto la moda de etiquetas en la ventana
            y_windows.append(tag)
        tags, freq = np.unique(np.array(y_windows), return_counts=True)
        for window in np.array(y_windows):
            tags, freq = np.unique(window, return_counts=True)
            y.append(tags[np.argmax(freq)])

        #windows_labels.append(labels_no_time[i:i+window_size])

    print("Hecho!\n")
    return np.array(x_windows), np.array(y)

def divide_dataset_into_windows(dataset, tags, window_size, moda):
    dataset_no_time = dataset[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    tags_no_time = tags[['Etiqueta']]
    num_windows = math.floor(len(dataset)/window_size)
    dataset_to_numpy = dataset_no_time.to_numpy()
    tags_to_numpy = tags_no_time.to_numpy()
    x_windows = []
    y = []

    for i in range(len(dataset_to_numpy)-window_size):
        elem = dataset_to_numpy[i:i+window_size]
        x_windows.append(elem)
    if moda is False:
        for i in range(len(tags_to_numpy)-window_size):
            tag = tags_to_numpy[i+window_size-1] #cogemos como etiqueta del conjunto la etiqueta del último elemento
            y.append(tag)
    else:
        y_windows = []
        for i in range(len(tags_to_numpy)-window_size):
            tag = tags_to_numpy[i:i+window_size] #cogemos como etiqueta del conjunto la moda de etiquetas en la ventana
            y_windows.append(tag)
        tags, freq = np.unique(np.array(y_windows), return_counts=True)
        for window in np.array(y_windows):
            tags, freq = np.unique(window, return_counts=True)
            y.append(tags[np.argmax(freq)])
    print("Hecho!\n")
    return np.array(x_windows), np.array(y)    

def divide_train_test(x, y, train_percent):
    tam_train = math.floor(len(x)*train_percent)

    return np.array(x[:tam_train]), np.array(y[:tam_train]), np.array(x[tam_train:]), np.array(y[tam_train:])

def create_model(input_dim, name_model_saved):
    model = Sequential()
    model.add(Input(shape=input_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    cp = ModelCheckpoint(name_model_saved, save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    return model, cp

def train_model(x_train, y_train, model, cp):
    model.fit(x_train, y_train, epochs=100, callbacks=[cp])

    return

def make_predictions(model, x_train, y_train, x_test, y_test, position_date):

    # clf_220617_bag = joblib.load('modelo_220617_bag.pkl') # Carga del modelo.

    model.score(x_train, y_train) #rendimiento del modelo

    y_preds = model.predict(x_test)
    f1_score_220617_bag = f1_score(y_test, y_preds, average='weighted')
    with open("/home/cmonedero/TFG/Perceptron/Norm/Perceptron_Results.txt", "w") as archivo_results:
        archivo_results.write("F1 score " + str(position_date) + " ---> " + str(f1_score_220617_bag) + "\n")

    return

def get_data(datos, tags):
    User1_data = pd.read_csv(datos).fillna(0)
    Label_New = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta'])
    
    print("Concatenando dataset con tags...")
    User1_data.reset_index(drop=True, inplace=True)
    Label_New.reset_index(drop=True, inplace=True)
    u1_data = pd.DataFrame(scaler.fit_transform(User1_data), columns=User1_data.columns)
    u1_data.reset_index(drop=True, inplace=True)
    Label_New.reset_index(drop=True, inplace=True) 
    df_User1_data = pd.concat([u1_data, Label_New['Etiqueta']], axis=1)
    
    print("Dividiendo en train y test...")
    x_train, x_test, y_train, y_test = train_test_split(u1_data[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']], Label_New['Etiqueta'], test_size=0.3)
    print("Tamaño del y_test: " + str(len(y_test)))

    return x_train, x_test, y_train, y_test

def Get_New_Labels(pred_bag, pred_hand, pred_hips):
    new_tag = []
    for bag, hand, hip in zip(pred_bag, pred_hand, pred_hips):
        if bag != hand != hip: #si son distintos se añade la etiqueta deducida por bag que es la de mejor rendimiento
            new_tag.append(bag)
        else: #si entra en esta condición es porque hay algún par de datos iguales
            if bag == hand == hip: #o bien los tres
                new_tag.append(bag)
            elif bag == hand and (bag != hip or hand != hip): #o bien los dos primeros
                new_tag.append(bag)
            elif bag == hip and (bag != hand or hip != hand): #el primero y elúltimo
                new_tag.append(bag)
            elif hand == hip and (hand != bag or hip != bag): #o los dos últimos
                new_tag.append(hand)
    return new_tag
