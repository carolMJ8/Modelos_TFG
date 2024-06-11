import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
scaler = StandardScaler()

def get_data(datos, tags):
    data = pd.read_csv(datos).fillna(0)
    labels = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta', 'Tipo Carretera'])
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data_norm.reset_index(drop=True, inplace=True)

    df = pd.concat([data, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    User1_data_no_time = df_reduced[[]]
    User1_data = df_reduced[['Tiempo', 'Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    Label_New = df_reduced[['Tiempo', 'Etiqueta']]

    print("Hecho!\n")
    return User1_data, Label_New

def create_windows(dataset, tags, window_size, num_windows):
    windows = TimeSeriesSplit(n_splits=3651, gap=250, max_train_size=window_size, test_size=500)
    all_splits = list(windows.split(dataset, tags))
    train, test = all_splits[0] #el primer split    
    print(dataset.iloc[train])
    print(dataset.iloc[test])
    #print(all_splits[0][0]) #aqui se guarda el primer split del dataset, la parte DE ENTRENAMIENTO
    #Tenemos que devolver un array el cual se componga de los 5 splits de train y otro con los 5 splits de test
    train_windows = []
    train_array = []
    test_windows = []
    test_array = []
    for i in range(len(all_splits)): #Aqui se construyen los arrays con los datos del dataset ya divididos en ventanas
        train, test = all_splits[i]
        train_windows.append(np.array(dataset.iloc[train]))
        train_array.append(np.array(dataset[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']].iloc[train]))
        test_windows.append(np.array(dataset.iloc[test]))
        test_array.append(np.array(dataset[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']].iloc[test]))

    #Ahora hay que etiquetar cada uno de los splits del dataset
    train_tag_times = []
    for i in range(len(train_windows)):
        slize = train_windows[i] #Cogemos la ventana que corresponda
        train_tag_times.append(slize[-1][0]) #Metemos en el array de tiempos el dato correspondiente

    test_tag_times = []
    for i in range(len(test_windows)):
        slize = test_windows[i] #Cogemos la ventana que corresponda
        test_tag_times.append(slize[-1][0]) #Metemos en el array de tiempos el dato correspondiente

    np_tags = np.array(tags)
    np_train_tag_times = np.array(train_tag_times)
    np_test_tag_times = np.array(test_tag_times)
    train_tags = []
    test_tags = []
    #Hacer el match de cada uno de los elementos del train_tag_times con el que sea de los tags
    #y meter en train_tags las etiquetas que sean
    for i in range(len(np_tags)):
        if np_tags[i][0] in np_train_tag_times:
            train_tags.append(np_tags[i][1])

    for i in range(len(np_tags)):
        if np_tags[i][0] in np_test_tag_times:
            test_tags.append(np_tags[i][1])

    print("Hecho!\n")
    return np.array(train_array), np.array(train_tags), np.array(test_array), np.array(test_tags)

def create_model(input_shape):
    #inputs = keras.Input(shape=input_shape)
    #x = inputs
    model = Sequential()
    model.add(LSTM(input_shape=input_shape, units=8, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.0, recurrent_dropout=0.0,
                    unroll=False, use_bias=True))
    model.add(Flatten())
    model.add(Dense(units=8, activation='softmax'))
    """    model = Sequential([
        Input(shape=(500,6)),
        LSTM(8, return_sequences=True),
        Flatten(),
        Dense(8, 'softmax')
    ])"""
    # model = Sequential()
    # model.add(InputLayer((500,6)))
    # model.add(LSTM(8))
    # model.add(Dense(8, 'softmax'))
    return model

def train_model(model, cp, x_train):
    model.fit(x_train, epochs=5, callbacks=[cp])

    return

def divide_dataset_into_windows(dataset, tags, window_size):
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
    for i in range(len(tags_to_numpy)-window_size):
        tag = tags_to_numpy[i+window_size-1] #cogemos como etiqueta del conjunto la etiqueta del último elemento
        y.append(tag)
        
    print("Hecho!\n")
    return np.array(x_windows), np.array(y)

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

def divide_train_test(x, y, train_percent):
    tam_train = math.floor(len(x)*train_percent)

    print("Hecho!\n")
    return np.array(x[:tam_train]), np.array(y[:tam_train]), np.array(x[tam_train:]), np.array(y[tam_train:])

def get_windows_from_dataset(datos, tags, window_size):
    User1_data = pd.read_csv(datos).fillna(0)
    Label_New = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta'])
    
    print("Concatenando dataset con tags...")
    User1_data.reset_index(drop=True, inplace=True)
    Label_New.reset_index(drop=True, inplace=True)
    u1_data = pd.DataFrame(scaler.fit_transform(User1_data), columns=User1_data.columns)
    u1_data.reset_index(drop=True, inplace=True)
    Label_New.reset_index(drop=True, inplace=True) 
    df_User1_data = pd.concat([u1_data, Label_New['Etiqueta']], axis=1)

    dataset_to_numpy = df_User1_data.to_numpy()

    print("Dividiendo en train y test...")
    x_train, x_test, y_train, y_test = train_test_split(u1_data[['Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']], Label_New['Etiqueta'], test_size=0.3)
    print("Tamaño del y_test: " + str(len(y_test)))

    return x_train, x_test, y_train, y_test
