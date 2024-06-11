import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import math
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

    df = pd.concat([data, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    User1_data_no_time = df_reduced[[]]
    User1_data = df_reduced[['Tiempo', 'Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    Label_New = df_reduced[['Tiempo', 'Etiqueta']]

    print("Hecho!\n")
    return User1_data, Label_New

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

    return np.array(x[:tam_train]), np.array(y[:tam_train]), np.array(x[tam_train:]), np.array(y[tam_train:])


def create_model(num_model):
    model = Sequential()
    if num_model == 1:
        model.add(InputLayer((500,6)))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(500,6)))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(500,6)))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(500,6)))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(500,6)))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(500,6)))
        model.add(Dropout(0.25))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100 ,'relu'))
        model.add(Dense(8, 'softmax'))
        model.summary()
    else:
        model.add(InputLayer((500,6)))
        model.add(Conv1D(100, 10, activation='relu'))
        model.add(Conv1D(100, 10, activation='relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(Conv1D(160, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(50 ,'relu'))
        model.add(Dense(8, 'softmax'))
        model.summary()       

    return model

def train_model(model, cp, x_train, y_train):
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    model.fit(x_train, y_train, validation_split=0.2, epochs=10, callbacks=[cp])

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
