import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

scaler = StandardScaler()

def get_data(datos, tags):
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

def multiheadattention(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = BatchNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # x = LayerNormalization(epsilon=1e-6)(res)
    # x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    # x = Dropout(dropout)(x)
    # x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res

def build_model_distributed(strategy, input_shape, head_size, num_heads, ff_dim, mlp_units, num_blocks, mlp_dropout, dropout):
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(num_blocks):
            x = multiheadattention(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        x = Dense(5, activation='relu')(x)
        x = Dropout(mlp_dropout)(x)
        
        outputs = Dense(8, 'softmax')(x)

        model = keras.Model(inputs, outputs)

        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.00001), metrics=[RootMeanSquaredError()])

    return model

def train_model_distributed(model, x, y, epochs, callbacks):
    model.fit(x, y, epochs=epochs, callbacks=callbacks)

    return

def build_model(input_shape, head_size, num_heads, ff_dim, num_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(num_blocks):
        x = multiheadattention(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Dense(10, activation='relu')(x)
    x = Dropout(mlp_dropout)(x)
    
    outputs = Dense(8, 'softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    return keras.Model(inputs, outputs)

def train_model(model, cp, x_train, y_train):
    model.fit(x_train, y_train, epochs=10, callbacks=[cp])

    return

def df_from_tags(y_tags):
    tags, freq = np.unique(np.array(y_tags), return_counts=True)
    props = []
    tuple = []
    for t, f in zip(tags, freq):
        tuple.append([t, f, (f/len(y_tags))*100])
    df = pd.DataFrame(tuple, columns=['tag', 'freq', 'prop'])
    return df

def create_diagrams(df, title, title2):
    fig, ax = plt.subplots()
    barras = ax.bar(df['tag'], df['freq'])
    for barra in barras:
        h = barra.get_height()
        ax.annotate(f'{h}', xy=(barra.get_x() + barra.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.set_title(title, loc="left", fontdict={'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    ax.set_xlabel('Etiqueta', fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    ax.set_ylabel('Frecuencia', fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    plt.show()

    #Grafico de sectores
    fig1, ax1 = plt.subplots()
    patches, _, _ = ax1.pie(df['prop'], autopct='%1.1f%%', labels=df['tag'])
    ax1.legend(handles=patches, labels=df['tag'].tolist(), loc='upper right')
    ax1.set_title(title2, loc="left", fontdict={'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    plt.show()

    return
