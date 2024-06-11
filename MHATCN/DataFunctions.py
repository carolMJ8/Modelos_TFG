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
from tcn import tcn

scaler = StandardScaler()

def process_data_for_paper_model(data, title):
    tam = 0
    with open(data, "r") as motion:
        for linea in motion:
            tam += 1

    acel_x = np.zeros(shape=tam)
    acel_y = np.zeros(shape=tam)
    acel_z = np.zeros(shape=tam)

    magnetometro_x =  np.zeros(shape=tam)
    magnetometro_y =  np.zeros(shape=tam)
    magnetometro_z =  np.zeros(shape=tam)

    giroscopio_x = np.zeros(shape=tam)
    giroscopio_y = np.zeros(shape=tam)
    giroscopio_z = np.zeros(shape=tam)

    presion = np.zeros(shape=tam)

    index = 0
    with open(data, "r") as motion:
        for linea in motion:
            linea_separada = linea.split()

            acel_x[index] = linea_separada[1]
            acel_y[index]  = linea_separada[2]
            acel_z[index]  = linea_separada[3]

            magnetometro_x[index]  =  linea_separada[7]
            magnetometro_y[index]  =  linea_separada[8]
            magnetometro_z[index]  =  linea_separada[9]

            giroscopio_x[index]  = linea_separada[4]
            giroscopio_y[index]  = linea_separada[5]
            giroscopio_z[index]  = linea_separada[6]

            presion[index]  = linea_separada[20]

            index += 1

    #dataframes con los valores de la aceleración lineal
    df_acel_x = pd.DataFrame(acel_x, columns=['Acelerometro en el eje X'])
    df_acel_y = pd.DataFrame(acel_y, columns=['Acelerometro en el eje Y'])
    df_acel_z = pd.DataFrame(acel_z, columns=['Acelerometro en el eje Z'])

    #dataframes con los valores del magnetometro
    df_magnetometro_x = pd.DataFrame(magnetometro_x, columns=['Magnetometro en el eje X'])
    df_magnetometro_y = pd.DataFrame(magnetometro_y, columns=['Magnetometro en el eje Y'])
    df_magnetometro_z = pd.DataFrame(magnetometro_z, columns=['Magnetometro en el eje Z'])

    #dataframes con los valores del giroscopio
    df_giroscopio_x = pd.DataFrame(giroscopio_x, columns=['Giroscopio en el eje X'])
    df_giroscopio_y = pd.DataFrame(giroscopio_y, columns=['Giroscopio en el eje Y'])
    df_giroscopio_z = pd.DataFrame(giroscopio_z, columns=['Giroscopio en el eje Z'])

    #dataframe con los valores de la presión
    df_presion = pd.DataFrame(presion, columns=['Presion'])

    df_acel_x = df_acel_x.reset_index(drop=True)
    df_acel_y = df_acel_y.reset_index(drop=True)
    df_acel_z = df_acel_z.reset_index(drop=True)
    df_magnetometro_x = df_magnetometro_x.reset_index(drop=True)
    df_magnetometro_y = df_magnetometro_y.reset_index(drop=True)
    df_magnetometro_z = df_magnetometro_z.reset_index(drop=True)
    df_giroscopio_x = df_giroscopio_x.reset_index(drop=True)
    df_giroscopio_y = df_giroscopio_y.reset_index(drop=True)
    df_giroscopio_z = df_giroscopio_z.reset_index(drop=True)
    df_presion = df_presion.reset_index(drop=True)

    DF = pd.concat([df_acel_x['Acelerometro en el eje X'], df_acel_y['Acelerometro en el eje Y'], df_acel_z['Acelerometro en el eje Z'], df_magnetometro_x['Magnetometro en el eje X'], df_magnetometro_y['Magnetometro en el eje Y'], df_magnetometro_z['Magnetometro en el eje Z'], df_giroscopio_x['Giroscopio en el eje X'], df_giroscopio_y['Giroscopio en el eje Y'], df_giroscopio_z['Giroscopio en el eje Z'], df_presion['Presion']], axis=1)
    DF.to_csv(title, header=['Acelerometro en el eje X', 'Acelerometro en el eje Y', 'Acelerometro en el eje Z', 'Magnetometro en el eje X', 'Magnetometro en el eje Y', 'Magnetometro en el eje Z', 'Giroscopio en el eje X', 'Giroscopio en el eje Y', 'Giroscopio en el eje Z', 'Presion'])

    return

def get_data(data, tags):
    data = pd.read_csv(data).fillna(0)
    labels = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta', 'Tipo Carretera'])
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data_norm.reset_index(drop=True, inplace=True)

    df = pd.concat([data_norm, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    data_final = df_reduced[['Tiempo', 'Acelerometro en el eje X', 'Acelerometro en el eje Y', 'Acelerometro en el eje Z', 'Magnetometro en el eje X', 'Magnetometro en el eje Y', 'Magnetometro en el eje Z', 'Giroscopio en el eje X', 'Giroscopio en el eje Y', 'Giroscopio en el eje Z', 'Presion']]
    label_final = df_reduced[['Tiempo', 'Etiqueta']]

    return data_final, label_final

def get_data_modules(data, tags):
    data = pd.read_csv(data).fillna(0)
    labels = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta', 'Tipo Carretera'])
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data_norm.reset_index(drop=True, inplace=True)

    df = pd.concat([data_norm, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    data_final = df_reduced[['Tiempo', 'Módulo Acelerómetro', 'Módulo Giroscopio', 'Módulo Magnetómetro', 'Módulo Orientación', 'Módulo Gravedad', 'Módulo Acel. Lineal']]
    label_final = df_reduced[['Tiempo', 'Etiqueta']]

    return data_final, label_final  

def get_arrays_from_data(data, tags):
    array_dict = {}

    acel_x = np.array(data['Acelerometro en el eje X'])
    df_acel_x = pd.DataFrame(acel_x, columns=['AcelX'])
    acel_y = np.array(data['Acelerometro en el eje Y'])
    df_acel_y = pd.DataFrame(acel_y, columns=['AcelY'])
    acel_z = np.array(data['Acelerometro en el eje Z'])
    df_acel_z = pd.DataFrame(acel_z, columns=['AcelZ'])

    magnetometro_x = np.array(data['Magnetometro en el eje X'])
    df_magnetometro_x = pd.DataFrame(magnetometro_x, columns=['MagX'])
    magnetometro_y = np.array(data['Magnetometro en el eje Y'])
    df_magnetometro_y = pd.DataFrame(magnetometro_y, columns=['MagY'])
    magnetometro_z = np.array(data['Magnetometro en el eje Z'])
    df_magnetometro_z = pd.DataFrame(magnetometro_z, columns=['MagZ'])

    giroscopio_x = np.array(data['Giroscopio en el eje X'])
    df_giroscopio_x = pd.DataFrame(giroscopio_x, columns=['GirosX'])
    giroscopio_y = np.array(data['Giroscopio en el eje Y'])
    df_giroscopio_y = pd.DataFrame(giroscopio_y, columns=['GirosY'])
    giroscopio_z = np.array(data['Giroscopio en el eje Z'])
    df_giroscopio_z = pd.DataFrame(giroscopio_z, columns=['GirosZ'])

    presion = np.array(data['Presion'])
    df_presion = pd.DataFrame(presion, columns=['Pres'])

    df = pd.concat([df_acel_x, df_acel_y, df_acel_z, df_magnetometro_x, df_magnetometro_y, df_magnetometro_z, df_giroscopio_x, df_giroscopio_y, df_giroscopio_z, df_presion], axis=1)

    array_dict['AcelX'] = acel_x
    array_dict['AcelY'] = acel_y
    array_dict['AcelZ'] = acel_z
    array_dict['MagX'] = magnetometro_x
    array_dict['MagY'] = magnetometro_y
    array_dict['MagZ'] = magnetometro_z
    array_dict['GirosX'] = giroscopio_x
    array_dict['GirosY'] = giroscopio_y
    array_dict['GirosZ'] = giroscopio_z
    array_dict['Pres'] = presion

    return df, array_dict

def get_arrays_from_data_modules(data, tags):
    array_dict = {}

    acel = np.array(data['Módulo Acelerómetro'])
    df_acel = pd.DataFrame(acel, columns=['Acel'])

    giros = np.array(data['Módulo Giroscopio'])
    df_giros = pd.DataFrame(giros, columns=['Giros'])

    mag = np.array(data['Módulo Magnetómetro'])
    df_mag = pd.DataFrame(mag, columns=['Mag'])

    ori = np.array(data['Módulo Orientación'])
    df_ori = pd.DataFrame(ori, columns=['Ori'])

    grav = np.array(data['Módulo Gravedad'])
    df_grav = pd.DataFrame(grav, columns=['Grav'])

    aceLineal = np.array(data['Módulo Acel. Lineal'])
    df_acel_lineal = pd.DataFrame(aceLineal, columns=['AceLineal'])

    df = pd.concat([df_acel, df_giros, df_mag, df_ori, df_grav, df_acel_lineal], axis=1)

    array_dict['Acel'] = acel
    array_dict['Giros'] = giros
    array_dict['Mag'] = mag
    array_dict['Ori'] = ori
    array_dict['Grav'] = grav
    array_dict['AceLineal'] = aceLineal

    return df, array_dict

def divide_data_arrays_into_windows(array_dict, labels, moda, window_size):
    acel_x = array_dict['AcelX'].tolist()
    acel_y = array_dict['AcelY'].tolist()
    acel_z = array_dict['AcelZ'].tolist()

    magnetometro_x = array_dict['MagX'].tolist()
    magnetometro_y = array_dict['MagY'].tolist()
    magnetometro_z = array_dict['MagZ'].tolist()

    giroscopio_x = array_dict['GirosX'].tolist()
    giroscopio_y = array_dict['GirosY'].tolist()
    giroscopio_z = array_dict['GirosZ'].tolist()

    presion = array_dict['Pres'].tolist()

    # dataset_to_numpy = dataset_no_time.to_numpy()
    labels_no_time = labels['Etiqueta'].to_numpy()

    tam_arrays = len(acel_x)

    windows_acel_x = []
    windows_acel_y = []
    windows_acel_z = []
    windows_magnetometro_x = []
    windows_magnetometro_y = []
    windows_magnetometro_z = []
    windows_giroscopio_x = []
    windows_giroscopio_y = []
    windows_giroscopio_z = []
    windows_presion = []

    windows_labels = []

    for i in range(0, tam_arrays - window_size, window_size):
        elem = acel_x[i:i+window_size]
        windows_acel_x.append(acel_x[i:i+window_size])
        windows_acel_y.append(acel_y[i:i+window_size])
        windows_acel_z.append(acel_z[i:i+window_size])
        windows_magnetometro_x.append(magnetometro_x[i:i+window_size])
        windows_magnetometro_y.append(magnetometro_y[i:i+window_size])
        windows_magnetometro_z.append(magnetometro_z[i:i+window_size])
        windows_giroscopio_x.append(giroscopio_x[i:i+window_size])
        windows_giroscopio_y.append(giroscopio_y[i:i+window_size])
        windows_giroscopio_z.append(giroscopio_z[i:i+window_size])
        windows_presion.append(presion[i:i+window_size])

    if moda is False:
        for i in range(0, tam_arrays - window_size, window_size):
            windows_labels.append(labels_no_time[i+window_size-1])
    else:
        y_windows = []
        for i in range(0, tam_arrays - window_size, window_size):
            tag = labels_no_time[i:i+window_size] #cogemos como etiqueta del conjunto la moda de etiquetas en la ventana
            y_windows.append(tag)
        tags, freq = np.unique(np.array(y_windows), return_counts=True)
        for window in np.array(y_windows):
            tags, freq = np.unique(window, return_counts=True)
            windows_labels.append(tags[np.argmax(freq)])

        #windows_labels.append(labels_no_time[i:i+window_size])

    array_dict_new = {}

    array_dict_new['AcelX'] = np.array(windows_acel_x)
    array_dict_new['AcelY'] = np.array(windows_acel_y)
    array_dict_new['AcelZ'] = np.array(windows_acel_z)
    array_dict_new['MagX'] = np.array(windows_magnetometro_x)
    array_dict_new['MagY'] = np.array(windows_magnetometro_y)
    array_dict_new['MagZ'] = np.array(windows_magnetometro_z)
    array_dict_new['GirosX'] = np.array(windows_giroscopio_x)
    array_dict_new['GirosY'] = np.array(windows_giroscopio_y)
    array_dict_new['GirosZ'] = np.array(windows_giroscopio_z)
    array_dict_new['Pres'] = np.array(windows_presion)

    return array_dict_new, np.array(windows_labels)

def divide_train_test(x, y, train_percent):
    tam_train = math.floor(len(x['AcelX'])*train_percent)

    acel_x = x['AcelX'].tolist()
    acel_y = x['AcelY'].tolist()
    acel_z = x['AcelZ'].tolist()

    magnetometro_x = x['MagX'].tolist()
    magnetometro_y = x['MagY'].tolist()
    magnetometro_z = x['MagZ'].tolist()

    giroscopio_x = x['GirosX'].tolist()
    giroscopio_y = x['GirosY'].tolist()
    giroscopio_z = x['GirosZ'].tolist()

    presion = x['Pres'].tolist()

    array_dict_train = {}
    array_dict_test = {}

    array_dict_train['AcelX'] = np.array(acel_x[:tam_train]).reshape(np.array(acel_x[:tam_train]).shape[0], np.array(acel_x[:tam_train]).shape[1], 1)
    array_dict_train['AcelY'] = np.array(acel_y[:tam_train]).reshape(np.array(acel_y[:tam_train]).shape[0], np.array(acel_y[:tam_train]).shape[1], 1)
    array_dict_train['AcelZ'] = np.array(acel_z[:tam_train]).reshape(np.array(acel_z[:tam_train]).shape[0], np.array(acel_z[:tam_train]).shape[1], 1)
    array_dict_train['MagX'] = np.array(magnetometro_x[:tam_train]).reshape(np.array(magnetometro_x[:tam_train]).shape[0], np.array(magnetometro_x[:tam_train]).shape[1], 1)
    array_dict_train['MagY'] = np.array(magnetometro_y[:tam_train]).reshape(np.array(magnetometro_y[:tam_train]).shape[0], np.array(magnetometro_y[:tam_train]).shape[1], 1)
    array_dict_train['MagZ'] = np.array(magnetometro_z[:tam_train]).reshape(np.array(magnetometro_z[:tam_train]).shape[0], np.array(magnetometro_z[:tam_train]).shape[1], 1)
    array_dict_train['GirosX'] = np.array(giroscopio_x[:tam_train]).reshape(np.array(giroscopio_x[:tam_train]).shape[0], np.array(giroscopio_x[:tam_train]).shape[1], 1)
    array_dict_train['GirosY'] = np.array(giroscopio_y[:tam_train]).reshape(np.array(giroscopio_y[:tam_train]).shape[0], np.array(giroscopio_y[:tam_train]).shape[1], 1)
    array_dict_train['GirosZ'] = np.array(giroscopio_z[:tam_train]).reshape(np.array(giroscopio_z[:tam_train]).shape[0], np.array(giroscopio_z[:tam_train]).shape[1], 1)
    array_dict_train['Pres'] = np.array(presion[:tam_train]).reshape(np.array(presion[:tam_train]).shape[0], np.array(presion[:tam_train]).shape[1], 1)

    array_dict_test['AcelX'] = np.array(acel_x[tam_train:]).reshape(np.array(acel_x[tam_train:]).shape[0], np.array(acel_x[tam_train:]).shape[1], 1)
    array_dict_test['AcelY'] = np.array(acel_y[tam_train:]).reshape(np.array(acel_y[tam_train:]).shape[0], np.array(acel_y[tam_train:]).shape[1], 1)
    array_dict_test['AcelZ'] = np.array(acel_z[tam_train:]).reshape(np.array(acel_z[tam_train:]).shape[0], np.array(acel_z[tam_train:]).shape[1], 1)
    array_dict_test['MagX'] = np.array(magnetometro_x[tam_train:]).reshape(np.array(magnetometro_x[tam_train:]).shape[0], np.array(magnetometro_x[tam_train:]).shape[1], 1)
    array_dict_test['MagY'] = np.array(magnetometro_y[tam_train:]).reshape(np.array(magnetometro_y[tam_train:]).shape[0], np.array(magnetometro_y[tam_train:]).shape[1], 1)
    array_dict_test['MagZ'] = np.array(magnetometro_z[tam_train:]).reshape(np.array(magnetometro_z[tam_train:]).shape[0], np.array(magnetometro_z[tam_train:]).shape[1], 1)
    array_dict_test['GirosX'] = np.array(giroscopio_x[tam_train:]).reshape(np.array(giroscopio_x[tam_train:]).shape[0], np.array(giroscopio_x[tam_train:]).shape[1], 1)
    array_dict_test['GirosY'] = np.array(giroscopio_y[tam_train:]).reshape(np.array(giroscopio_y[tam_train:]).shape[0], np.array(giroscopio_y[tam_train:]).shape[1], 1)
    array_dict_test['GirosZ'] = np.array(giroscopio_z[tam_train:]).reshape(np.array(giroscopio_z[tam_train:]).shape[0], np.array(giroscopio_z[tam_train:]).shape[1], 1)
    array_dict_test['Pres'] = np.array(presion[tam_train:]).reshape(np.array(presion[tam_train:]).shape[0], np.array(presion[tam_train:]).shape[1], 1)  


    return array_dict_train, np.array(y[:tam_train]), array_dict_test, np.array(y[tam_train:])

def create_train_test_tensors(x_train, x_test, y_train, y_test):
    acel_x_train = x_train['AcelX'].tolist()
    acel_y_train = x_train['AcelY'].tolist()
    acel_z_train = x_train['AcelZ'].tolist()

    magnetometro_x_train = x_train['MagX'].tolist()
    magnetometro_y_train = x_train['MagY'].tolist()
    magnetometro_z_train = x_train['MagZ'].tolist()

    giroscopio_x_train = x_train['GirosX'].tolist()
    giroscopio_y_train = x_train['GirosY'].tolist()
    giroscopio_z_train = x_train['GirosZ'].tolist()

    presion_train = x_train['Pres'].tolist()

    acel_x_test = x_test['AcelX'].tolist()
    acel_y_test = x_test['AcelY'].tolist()
    acel_z_test = x_test['AcelZ'].tolist()

    magnetometro_x_test = x_test['MagX'].tolist()
    magnetometro_y_test = x_test['MagY'].tolist()
    magnetometro_z_test = x_test['MagZ'].tolist()

    giroscopio_x_test = x_test['GirosX'].tolist()
    giroscopio_y_test = x_test['GirosY'].tolist()
    giroscopio_z_test = x_test['GirosZ'].tolist()

    presion_test = x_test['Pres'].tolist()

    dict_train_tensors = {}
    dict_test_tensors = {}

    dict_train_tensors['AcelX'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_x_train), tf.constant(y_train)))
    dict_train_tensors['AcelY'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_y_train), tf.constant(y_train)))
    dict_train_tensors['AcelZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_z_train), tf.constant(y_train)))
    dict_train_tensors['MagX'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_x_train), tf.constant(y_train)))
    dict_train_tensors['MagY'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_y_train), tf.constant(y_train)))
    dict_train_tensors['MagZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_z_train), tf.constant(y_train)))
    dict_train_tensors['GirosX'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_x_train), tf.constant(y_train)))
    dict_train_tensors['GirosY'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_y_train), tf.constant(y_train)))
    dict_train_tensors['GirosZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_z_train), tf.constant(y_train)))
    dict_train_tensors['Pres'] = tf.data.Dataset.from_tensor_slices((tf.constant(presion_train), tf.constant(y_train)))

    # dict_train_tensors['AcelX'] = (tf.constant(acel_x_train), tf.constant(y_train))
    # dict_train_tensors['AcelY'] = (tf.constant(acel_y_train), tf.constant(y_train))
    # dict_train_tensors['AcelZ'] = (tf.constant(acel_z_train), tf.constant(y_train))
    # dict_train_tensors['MagX'] = (tf.constant(magnetometro_x_train), tf.constant(y_train))
    # dict_train_tensors['MagY'] = (tf.constant(magnetometro_y_train), tf.constant(y_train))
    # dict_train_tensors['MagZ'] = (tf.constant(magnetometro_z_train), tf.constant(y_train))
    # dict_train_tensors['GirosX'] = (tf.constant(giroscopio_x_train), tf.constant(y_train))
    # dict_train_tensors['GirosY'] = (tf.constant(giroscopio_y_train), tf.constant(y_train))
    # dict_train_tensors['GirosZ'] = (tf.constant(giroscopio_z_train), tf.constant(y_train))
    # dict_train_tensors['Pres'] = (tf.constant(presion_train), tf.constant(y_train))

    train = batch_tensors(dict_train_tensors)

    dict_test_tensors['AcelX'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_x_test), tf.constant(y_test)))
    dict_test_tensors['AcelY'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_y_test), tf.constant(y_test)))
    dict_test_tensors['AcelZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(acel_z_test), tf.constant(y_test)))
    dict_test_tensors['MagX'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_x_test), tf.constant(y_test)))
    dict_test_tensors['MagY'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_y_test), tf.constant(y_test)))
    dict_test_tensors['MagZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(magnetometro_z_test), tf.constant(y_test)))
    dict_test_tensors['GirosX'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_x_test), tf.constant(y_test)))
    dict_test_tensors['GirosY'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_y_test), tf.constant(y_test)))
    dict_test_tensors['GirosZ'] = tf.data.Dataset.from_tensor_slices((tf.constant(giroscopio_z_test), tf.constant(y_test)))
    dict_test_tensors['Pres'] = tf.data.Dataset.from_tensor_slices((tf.constant(presion_test), tf.constant(y_test)))

    test = batch_tensors(dict_test_tensors)

    return train, test


def batch_tensors(dict):
    BATCH_SIZE=32
    dict_new = {}

    acel_x = dict['AcelX']
    acel_y = dict['AcelY']
    acel_z = dict['AcelZ']

    magnetometro_x = dict['MagX']
    magnetometro_y = dict['MagY']
    magnetometro_z = dict['MagZ']

    giroscopio_x = dict['GirosX']
    giroscopio_y = dict['GirosY']
    giroscopio_z = dict['GirosZ']

    presion = dict['Pres']

    dict_new['AcelX'] = acel_x.cache().batch(BATCH_SIZE)
    dict_new['AcelY'] = acel_y.cache().batch(BATCH_SIZE)
    dict_new['AcelZ'] = acel_z.cache().batch(BATCH_SIZE)
    
    dict_new['MagX'] = magnetometro_x.cache().batch(BATCH_SIZE)
    dict_new['MagY'] = magnetometro_y.cache().batch(BATCH_SIZE)
    dict_new['MagZ'] = magnetometro_z.cache().batch(BATCH_SIZE)
    
    dict_new['GirosX'] = giroscopio_x.cache().batch(BATCH_SIZE)
    dict_new['GirosY'] = giroscopio_y.cache().batch(BATCH_SIZE)
    dict_new['GirosZ'] = giroscopio_z.cache().batch(BATCH_SIZE)
    
    dict_new['Pres'] = presion.cache().batch(BATCH_SIZE)

    return dict_new

def create_model(input_shape):
    # Creamos los modelos para las 3 primeras capas para cada uno de los 10 tensores de entrada
    # La variable first_10_outputs es la concatenación de los 10 primeros modelos de entrada
    # en una única salida
    first_models_concat, inputs = create_10_first_models(input_shape)
    
    x = first_models_concat
    print(f'Capa de MHA --> {first_models_concat.shape}')
    # 4ª capa: CNN con kernel size de 64 y pool size de 2
    x = Conv1D(activation='relu', kernel_size=1, filters=64)(x)
    print(f'Capa de convolución --> {x.shape}')

    # Max pooling y global average pooling entre capas 4ª y 5ª
    x = MaxPool1D(pool_size=2, padding='valid')(x)
    print(f'Capa de MaxPooling --> {x.shape}')
    x = GlobalAveragePooling1D()(x)
    print(f'Capa de GAPooling --> {x.shape}')

    # Capa de flatten antes de la FC
    # x = Flatten()(x)
    # print(f'Capa de Flatten --> {x.shape}')

    #5ª capa: FC con 8 neuronas (número de características) y activación softmax
    outputs = Dense(8, 'softmax')(x)
    print(f'Capa de salida de la red --> {outputs.shape}')

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

    return model

def create_first_layers_model(input_shape, para_print):
    # 1ª capa: input layer con input shape (32, 500, 1)
    if para_print is True: print(f'Input shape {input_shape}')
    inputs = keras.Input(shape=input_shape)
    if para_print is True: print(f'Capa input --> {inputs.shape}')
    x = inputs

    # 2ª capa: TCN con 32 como número de filtros
    x = tcn.TCN(input_shape=input_shape,
            use_skip_connections=False,
            use_batch_norm=False,
            use_weight_norm=False,
            use_layer_norm=False,
            return_sequences=True,
            nb_filters=32,
            padding='causal')(x)
    if para_print is True: print(f'Capa TCN ---> {x.shape}')
    
    # Max pooling layer entre capas 2ª y 3ª
    x = MaxPool1D(pool_size=2, padding='valid')(x)
    if para_print is True: print(f'Capa MaxPooling ---> {x.shape}')

    # 3ª capa: MHA 
    x = BatchNormalization(epsilon=1e-6)(x)
    if para_print is True: print(f'Capa de normalizacion ---> {x.shape}')
    output = MultiHeadAttention(key_dim=4, num_heads=1, dropout=0.25 )(x, x)
    if para_print is True: print(f'Capa multi atención ---> {output.shape}')


    # Este trozo de código únicamente sirve para ver si las primeras capas de la red funcionan
    # # Capa de Flatten
    # x = Flatten()(x)
    # if para_print is True: print(f'Capa de Flatten ---> {x.shape}')

    # # Capa de salida
    # outputs = Dense(8, activation='softmax')(x)
    # if para_print is True: print(f'Capa de salida ---> {outputs.shape}')

    # model = keras.Model(inputs, outputs)
    # model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=[RootMeanSquaredError()])

    return output, inputs

def create_10_first_models(input_shape):

    output_acel_x, input_acel_x  = create_first_layers_model(input_shape, True)
    output_acel_y, input_acel_y  = create_first_layers_model(input_shape, False)
    output_acel_z, input_acel_z  = create_first_layers_model(input_shape, False)

    output_mag_x, input_mag_x  = create_first_layers_model(input_shape, False)
    output_mag_y, input_mag_y  = create_first_layers_model(input_shape, False)
    output_mag_z, input_mag_z  = create_first_layers_model(input_shape, False)

    output_giros_x, input_giros_x  = create_first_layers_model(input_shape, False)
    output_giros_y, input_giros_y  = create_first_layers_model(input_shape, False)
    output_giros_z, input_giros_z  = create_first_layers_model(input_shape, False)

    output_pres, input_pres  = create_first_layers_model(input_shape, False)

    inputs = [input_acel_x, input_acel_y, input_acel_z, input_mag_x, input_mag_y, input_mag_z, input_giros_x, input_giros_y, input_giros_z, input_pres]

    output_layer = Concatenate()([output_acel_x, output_acel_y, output_acel_z, output_mag_x, output_mag_y, output_mag_z, output_giros_x, output_giros_y, output_giros_z, output_pres])

    return output_layer, inputs

def train_first_layers_model(model, checkpoint_filepath, data):
    cp = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, save_freq='epoch')
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    model.fit(data, epochs=100, callbacks=cp)

def train_10_first_models(train_dict, data):

    train_first_layers_model(train_dict['AcelX'], '/home/cmonedero/TFG/MHATCN/Models/checkpointAcelX.model.keras', data['AcelX'])
    # train_first_layers_model(train_dict['AcelY'], '/home/cmonedero/TFG/MHATCN/Models/checkpointAcelY.model.keras', data['AcelY'])
    # train_first_layers_model(train_dict['AcelZ'], '/home/cmonedero/TFG/MHATCN/Models/checkpointAcelZ.model.keras', data['AcelZ'])

    # train_first_layers_model(train_dict['MagX'], '/home/cmonedero/TFG/MHATCN/Models/checkpointMagX.model.keras', data['MagX'])
    # train_first_layers_model(train_dict['MagY'], '/home/cmonedero/TFG/MHATCN/Models/checkpointMagY.model.keras', data['MagY'])
    # train_first_layers_model(train_dict['MagZ'], '/home/cmonedero/TFG/MHATCN/Models/checkpointMagZ.model.keras', data['MagZ'])

    # train_first_layers_model(train_dict['GirosX'], '/home/cmonedero/TFG/MHATCN/Models/checkpointGirosX.model.keras', data['GirosX'])
    # train_first_layers_model(train_dict['GirosY'], '/home/cmonedero/TFG/MHATCN/Models/checkpointGirosY.model.keras', data['GirosY'])
    # train_first_layers_model(train_dict['GirosZ'], '/home/cmonedero/TFG/MHATCN/Models/checkpointGirosZ.model.keras', data['GirosZ'])    

    # train_first_layers_model(train_dict['Pres'], '/home/cmonedero/TFG/MHATCN/Models/checkpointPres.model.keras', data['Pres'])  

    return train_dict

def concatenate_10_models_outputs(dict_outputs):

    output_acel_x = dict_outputs['AcelX']
    output_acel_y = dict_outputs['AcelY']
    output_acel_z = dict_outputs['AcelZ']

    output_mag_x = dict_outputs['MagX']
    output_mag_y = dict_outputs['MagY']
    output_mag_z = dict_outputs['MagZ']

    output_giros_x = dict_outputs['GirosX']
    output_giros_y = dict_outputs['GirosY']
    output_giros_z = dict_outputs['GirosZ']

    output_pres = dict_outputs['Pres']

    list_outputs = [output_acel_x, output_acel_y, output_acel_z, output_mag_x, output_mag_y, output_mag_z, output_giros_x, output_giros_y, output_giros_z, output_pres]

    return tf.concat(list_outputs, axis=0)

def get_data_and_create_windows(data_path, labels_paths, less_data=False, num_data=0, title=None):
    raw_data, labels = get_data(data_path, labels_paths)
    
    if less_data is True and num_data != 0:
        raw_data = raw_data[num_data]
        labels = labels[num_data]

    df_arrays, dict_arrays = get_arrays_from_data(raw_data, labels)

    windows_dict, windows_labels = divide_data_arrays_into_windows(df_arrays, labels, True, 500)

    x_train, y_train, x_test, y_test = divide_train_test(windows_dict, windows_labels, 0.7)

    print(f'--INFO DE LOS DATOS {title} --')
    print(f'Tamaño dataset: {len(raw_data)}')
    print("Numero de ventanas de entrenamiento: ", len(x_train['AcelX']))
    print("Forma de las ventanas de entrenamiento: datos -->", x_train['AcelX'].shape, "; labels --> ", y_train.shape)
    print("Numero de ventanas de pruebas: ", len(x_test['AcelX']))
    print("Forma de las ventanas de pruebas: datos --> ", x_test['AcelX'].shape, "; labels --> ", y_test.shape)
    print(f'\n\n')

    return x_train, y_train, x_test, y_test

def get_data_and_create_windows_modules(data_path, labels_paths, less_data=False, num_data=0, title=None):
    raw_data, labels = get_data_modules(data_path, labels_paths)
    
    if less_data is True and num_data != 0:
        raw_data = raw_data[num_data]
        labels = labels[num_data]

    df_arrays, dict_arrays = get_arrays_from_data(raw_data, labels)

    windows_dict, windows_labels = divide_data_arrays_into_windows(df_arrays, labels, True, 500)

    x_train, y_train, x_test, y_test = divide_train_test(windows_dict, windows_labels, 0.7)

    print(f'--INFO DE LOS DATOS {title} --')
    print(f'Tamaño dataset: {len(raw_data)}')
    print("Numero de ventanas de entrenamiento: ", len(x_train['AcelX']))
    print("Forma de las ventanas de entrenamiento: datos -->", x_train['AcelX'].shape, "; labels --> ", y_train.shape)
    print("Numero de ventanas de pruebas: ", len(x_test['AcelX']))
    print("Forma de las ventanas de pruebas: datos --> ", x_test['AcelX'].shape, "; labels --> ", y_test.shape)
    print(f'\n\n')

    return x_train, y_train, x_test, y_test

def create_full_train_test_arrays(x_train_user1_220617_hips, x_test_user1_220617_hips,
                                   x_train_user1_260617_hips, x_test_user1_260617_hips,
                                   x_train_user2_180717_hand, x_test_user2_180717_hand,
                                   x_train_user3_140617_hips, x_test_user3_140617_hips,
                                   x_train_user3_070717_hips, x_test_user3_070717_hips,
                                   y_train_user1_220617, y_train_user1_260617, y_train_user2_180717, y_train_user3_140617, y_train_user3_070717,
                                   y_test_user1_220617, y_test_user1_260617, y_test_user2_180717, y_test_user3_140617, y_test_user3_070717):
    # ---------------- ACELEROMETRO EN X TRAIN
    acelx_train_user1_220617_hips = x_train_user1_220617_hips['AcelX']
    acelx_train_user1_260617_hips = x_train_user1_260617_hips['AcelX']
    acelx_train_user2_180717_hand = x_train_user2_180717_hand['AcelX']
    acelx_train_user3_140617_hips = x_train_user3_140617_hips['AcelX']
    acelx_train_user3_070717_hips = x_train_user3_070717_hips['AcelX']

    # ---------------- ACELEROMETRO EN X TEST
    acelx_test_user1_220617_hips = x_test_user1_220617_hips['AcelX']
    acelx_test_user1_260617_hips = x_test_user1_260617_hips['AcelX']
    acelx_test_user2_180717_hand = x_test_user2_180717_hand['AcelX']
    acelx_test_user3_140617_hips = x_test_user3_140617_hips['AcelX']
    acelx_test_user3_070717_hips = x_test_user3_070717_hips['AcelX']


    # ---------------- ACELEROMETRO EN Y TRAIN
    acely_train_user1_220617_hips = x_train_user1_220617_hips['AcelY']
    acely_train_user1_260617_hips = x_train_user1_260617_hips['AcelY']
    acely_train_user2_180717_hand = x_train_user2_180717_hand['AcelY']
    acely_train_user3_140617_hips = x_train_user3_140617_hips['AcelY']
    acely_train_user3_070717_hips = x_train_user3_070717_hips['AcelY']

    # ---------------- ACELEROMETRO EN Y TEST
    acely_test_user1_220617_hips = x_test_user1_220617_hips['AcelY']
    acely_test_user1_260617_hips = x_test_user1_260617_hips['AcelY']
    acely_test_user2_180717_hand = x_test_user2_180717_hand['AcelY']
    acely_test_user3_140617_hips = x_test_user3_140617_hips['AcelY']
    acely_test_user3_070717_hips = x_test_user3_070717_hips['AcelY']


    # ---------------- ACELEROMETRO EN Z TRAIN
    acelz_train_user1_220617_hips = x_train_user1_220617_hips['AcelZ']
    acelz_train_user1_260617_hips = x_train_user1_260617_hips['AcelZ']
    acelz_train_user2_180717_hand = x_train_user2_180717_hand['AcelZ']
    acelz_train_user3_140617_hips = x_train_user3_140617_hips['AcelZ']
    acelz_train_user3_070717_hips = x_train_user3_070717_hips['AcelZ']

    # ---------------- ACELEROMETRO EN Z TEST
    acelz_test_user1_220617_hips = x_test_user1_220617_hips['AcelZ']
    acelz_test_user1_260617_hips = x_test_user1_260617_hips['AcelZ']
    acelz_test_user2_180717_hand = x_test_user2_180717_hand['AcelZ']
    acelz_test_user3_140617_hips = x_test_user3_140617_hips['AcelZ']
    acelz_test_user3_070717_hips = x_test_user3_070717_hips['AcelZ']


    # ---------------- MAGNETOMETRO EN X TRAIN
    magx_train_user1_220617_hips = x_train_user1_220617_hips['MagX']
    magx_train_user1_260617_hips = x_train_user1_260617_hips['MagX']
    magx_train_user2_180717_hand = x_train_user2_180717_hand['MagX']
    magx_train_user3_140617_hips = x_train_user3_140617_hips['MagX']
    magx_train_user3_070717_hips = x_train_user3_070717_hips['MagX']

    # ---------------- MAGNETOMETRO EN X TEST
    magx_test_user1_220617_hips = x_test_user1_220617_hips['MagX']
    magx_test_user1_260617_hips = x_test_user1_260617_hips['MagX']
    magx_test_user2_180717_hand = x_test_user2_180717_hand['MagX']
    magx_test_user3_140617_hips = x_test_user3_140617_hips['MagX']
    magx_test_user3_070717_hips = x_test_user3_070717_hips['MagX']


    # ---------------- MAGNETOMETRO EN Y TRAIN
    magy_train_user1_220617_hips = x_train_user1_220617_hips['MagY']
    magy_train_user1_260617_hips = x_train_user1_260617_hips['MagY']
    magy_train_user2_180717_hand = x_train_user2_180717_hand['MagY']
    magy_train_user3_140617_hips = x_train_user3_140617_hips['MagY']
    magy_train_user3_070717_hips = x_train_user3_070717_hips['MagY']

    # ---------------- MAGNETOMETRO EN Y TEST
    magy_test_user1_220617_hips = x_test_user1_220617_hips['MagY']
    magy_test_user1_260617_hips = x_test_user1_260617_hips['MagY']
    magy_test_user2_180717_hand = x_test_user2_180717_hand['MagY']
    magy_test_user3_140617_hips = x_test_user3_140617_hips['MagY']
    magy_test_user3_070717_hips = x_test_user3_070717_hips['MagY']


    # ---------------- MAGNETOMETRO EN Z TRAIN
    magz_train_user1_220617_hips = x_train_user1_220617_hips['MagZ']
    magz_train_user1_260617_hips = x_train_user1_260617_hips['MagZ']
    magz_train_user2_180717_hand = x_train_user2_180717_hand['MagZ']
    magz_train_user3_140617_hips = x_train_user3_140617_hips['MagZ']
    magz_train_user3_070717_hips = x_train_user3_070717_hips['MagZ']

    # ---------------- MAGNETOMETRO EN Z TEST
    magz_test_user1_220617_hips = x_test_user1_220617_hips['MagZ']
    magz_test_user1_260617_hips = x_test_user1_260617_hips['MagZ']
    magz_test_user2_180717_hand = x_test_user2_180717_hand['MagZ']
    magz_test_user3_140617_hips = x_test_user3_140617_hips['MagZ']
    magz_test_user3_070717_hips = x_test_user3_070717_hips['MagZ']


    # ---------------- GIROSCOPIO EN X TRAIN
    girosx_train_user1_220617_hips = x_train_user1_220617_hips['GirosX']
    girosx_train_user1_260617_hips = x_train_user1_260617_hips['GirosX']
    girosx_train_user2_180717_hand = x_train_user2_180717_hand['GirosX']
    girosx_train_user3_140617_hips = x_train_user3_140617_hips['GirosX']
    girosx_train_user3_070717_hips = x_train_user3_070717_hips['GirosX']

    # ---------------- GIROSCOPIO EN X TEST
    girosx_test_user1_220617_hips = x_test_user1_220617_hips['GirosX']
    girosx_test_user1_260617_hips = x_test_user1_260617_hips['GirosX']
    girosx_test_user2_180717_hand = x_test_user2_180717_hand['GirosX']
    girosx_test_user3_140617_hips = x_test_user3_140617_hips['GirosX']
    girosx_test_user3_070717_hips = x_test_user3_070717_hips['GirosX']


    # ---------------- GIROSCOPIO EN Y TRAIN
    girosy_train_user1_220617_hips = x_train_user1_220617_hips['GirosY']
    girosy_train_user1_260617_hips = x_train_user1_260617_hips['GirosY']
    girosy_train_user2_180717_hand = x_train_user2_180717_hand['GirosY']
    girosy_train_user3_140617_hips = x_train_user3_140617_hips['GirosY']
    girosy_train_user3_070717_hips = x_train_user3_070717_hips['GirosY']

    # ---------------- GIROSCOPIO EN Y TEST
    girosy_test_user1_220617_hips = x_test_user1_220617_hips['GirosY']
    girosy_test_user1_260617_hips = x_test_user1_260617_hips['GirosY']
    girosy_test_user2_180717_hand = x_test_user2_180717_hand['GirosY']
    girosy_test_user3_140617_hips = x_test_user3_140617_hips['GirosY']
    girosy_test_user3_070717_hips = x_test_user3_070717_hips['GirosY']


    # ---------------- GIROSCOPIO EN Z TRAIN
    girosz_train_user1_220617_hips = x_train_user1_220617_hips['GirosZ']
    girosz_train_user1_260617_hips = x_train_user1_260617_hips['GirosZ']
    girosz_train_user2_180717_hand = x_train_user2_180717_hand['GirosZ']
    girosz_train_user3_140617_hips = x_train_user3_140617_hips['GirosZ']
    girosz_train_user3_070717_hips = x_train_user3_070717_hips['GirosZ']

    # ---------------- GIROSCOPIO EN Z TEST
    girosz_test_user1_220617_hips = x_test_user1_220617_hips['GirosZ']
    girosz_test_user1_260617_hips = x_test_user1_260617_hips['GirosZ']
    girosz_test_user2_180717_hand = x_test_user2_180717_hand['GirosZ']
    girosz_test_user3_140617_hips = x_test_user3_140617_hips['GirosZ']
    girosz_test_user3_070717_hips = x_test_user3_070717_hips['GirosZ']


    # ---------------- PRESION TRAIN
    pres_train_user1_220617_hips = x_train_user1_220617_hips['Pres']
    pres_train_user1_260617_hips = x_train_user1_260617_hips['Pres']
    pres_train_user2_180717_hand = x_train_user2_180717_hand['Pres']
    pres_train_user3_140617_hips = x_train_user3_140617_hips['Pres']
    pres_train_user3_070717_hips = x_train_user3_070717_hips['Pres']

    # ---------------- PRESION TEST
    pres_test_user1_220617_hips = x_test_user1_220617_hips['Pres']
    pres_test_user1_260617_hips = x_test_user1_260617_hips['Pres']
    pres_test_user2_180717_hand = x_test_user2_180717_hand['Pres']
    pres_test_user3_140617_hips = x_test_user3_140617_hips['Pres']
    pres_test_user3_070717_hips = x_test_user3_070717_hips['Pres']

    acelx_train_array = [acelx_train_user3_140617_hips, acelx_train_user1_220617_hips, acelx_train_user1_260617_hips, acelx_train_user3_070717_hips, acelx_train_user2_180717_hand]

    acelx_test_array = [acelx_test_user3_140617_hips, acelx_test_user1_220617_hips, acelx_test_user1_260617_hips, acelx_test_user3_070717_hips, acelx_test_user2_180717_hand]
    
    acely_train_array = [acely_train_user3_140617_hips, acely_train_user1_220617_hips, acely_train_user1_260617_hips, acely_train_user3_070717_hips, acely_train_user2_180717_hand]

    acely_test_array = [acely_test_user3_140617_hips, acely_test_user1_220617_hips, acely_test_user1_260617_hips, acely_test_user3_070717_hips, acely_test_user2_180717_hand]

    acelz_train_array = [acelz_train_user3_140617_hips, acelz_train_user1_220617_hips, acelz_train_user1_260617_hips, acelz_train_user3_070717_hips, acelz_train_user2_180717_hand]

    acelz_test_array = [acelz_test_user3_140617_hips, acelz_test_user1_220617_hips, acelz_test_user1_260617_hips, acelz_test_user3_070717_hips, acelz_test_user2_180717_hand]

    magx_train_array = [magx_train_user3_140617_hips, magx_train_user1_220617_hips, magx_train_user1_260617_hips, magx_train_user3_070717_hips, magx_train_user2_180717_hand]

    magx_test_array = [magx_test_user3_140617_hips, magx_test_user1_220617_hips, magx_test_user1_260617_hips, magx_test_user3_070717_hips, magx_test_user2_180717_hand]

    magy_train_array = [magy_train_user3_140617_hips, magy_train_user1_220617_hips, magy_train_user1_260617_hips, magy_train_user3_070717_hips, magy_train_user2_180717_hand]

    magy_test_array = [magy_test_user3_140617_hips, magy_test_user1_220617_hips, magy_test_user1_260617_hips, magy_test_user3_070717_hips, magy_test_user2_180717_hand]

    magz_train_array = [magz_train_user3_140617_hips, magz_train_user1_220617_hips, magz_train_user1_260617_hips, magz_train_user3_070717_hips, magz_train_user2_180717_hand]

    magz_test_array = [magz_test_user3_140617_hips, magz_test_user1_220617_hips, magz_test_user1_260617_hips, magz_test_user3_070717_hips, magz_test_user2_180717_hand]

    girosx_train_array = [girosx_train_user3_140617_hips, girosx_train_user1_220617_hips, girosx_train_user1_260617_hips, girosx_train_user3_070717_hips, girosx_train_user2_180717_hand]

    girosx_test_array = [girosx_test_user3_140617_hips, girosx_test_user1_220617_hips, girosx_test_user1_260617_hips, girosx_test_user3_070717_hips, girosx_test_user2_180717_hand]

    girosy_train_array = [girosy_train_user3_140617_hips, girosy_train_user1_220617_hips, girosy_train_user1_260617_hips, girosy_train_user3_070717_hips, girosy_train_user2_180717_hand]

    girosy_test_array = [girosy_test_user3_140617_hips, girosy_test_user1_220617_hips, girosy_test_user1_260617_hips, girosy_test_user3_070717_hips, girosy_test_user2_180717_hand]

    girosz_train_array = [girosz_train_user3_140617_hips, girosz_train_user1_220617_hips, girosz_train_user1_260617_hips, girosz_train_user3_070717_hips, girosz_train_user2_180717_hand]

    girosz_test_array = [girosz_test_user3_140617_hips, girosz_test_user1_220617_hips, girosz_test_user1_260617_hips, girosz_test_user3_070717_hips, girosz_test_user2_180717_hand]

    pres_train_array = [pres_train_user3_140617_hips, pres_train_user1_220617_hips, pres_train_user1_260617_hips, pres_train_user3_070717_hips, pres_train_user2_180717_hand]

    pres_test_array = [pres_test_user3_140617_hips, pres_test_user1_220617_hips, pres_test_user1_260617_hips, pres_test_user3_070717_hips, pres_test_user2_180717_hand]

    y_train_array = [y_train_user3_140617, y_train_user1_220617, y_train_user1_260617, y_train_user3_070717, y_train_user2_180717]

    y_test_array = [y_test_user3_140617, y_test_user1_220617, y_test_user1_260617, y_test_user3_070717, y_test_user2_180717]

    acelx_train = np.concatenate(acelx_train_array, axis=0)
    acely_train = np.concatenate(acely_train_array, axis=0)
    acelz_train = np.concatenate(acelz_train_array, axis=0)

    magx_train = np.concatenate(magx_train_array, axis=0)
    magy_train = np.concatenate(magy_train_array, axis=0)
    magz_train = np.concatenate(magz_train_array, axis=0)

    girosx_train = np.concatenate(girosx_train_array, axis=0)
    girosy_train = np.concatenate(girosy_train_array, axis=0)
    girosz_train = np.concatenate(girosz_train_array, axis=0)

    pres_train = np.concatenate(pres_train_array, axis=0)

    acelx_test = np.concatenate(acelx_test_array, axis=0)
    acely_test = np.concatenate(acely_test_array, axis=0)
    acelz_test = np.concatenate(acelz_test_array, axis=0)

    magx_test = np.concatenate(magx_test_array, axis=0)
    magy_test = np.concatenate(magy_test_array, axis=0)
    magz_test = np.concatenate(magz_test_array, axis=0)

    girosx_test = np.concatenate(girosx_test_array, axis=0)
    girosy_test = np.concatenate(girosy_test_array, axis=0)
    girosz_test = np.concatenate(girosz_test_array, axis=0)

    pres_test = np.concatenate(pres_test_array, axis=0)

    y_train = np.concatenate(y_train_array, axis=0)
    y_test = np.concatenate(y_test_array, axis=0)

    x_train = {}
    x_test = {}
    x_train['AcelX'] = acelx_train
    x_train['AcelY'] = acely_train
    x_train['AcelZ'] = acelz_train
    x_train['MagX'] = magx_train
    x_train['MagY'] = magy_train
    x_train['MagZ'] = magz_train
    x_train['GirosX'] = girosx_train
    x_train['GirosY'] = girosy_train
    x_train['GirosZ'] = girosz_train
    x_train['Pres'] = pres_train
    
    x_test['AcelX'] = acelx_test
    x_test['AcelY'] = acely_test
    x_test['AcelZ'] = acelz_test
    x_test['MagX'] = magx_test
    x_test['MagY'] = magy_test
    x_test['MagZ'] = magz_test
    x_test['GirosX'] = girosx_test
    x_test['GirosY'] = girosy_test
    x_test['GirosZ'] = girosz_test
    x_test['Pres'] = pres_test

    return x_train, x_test, y_train, y_test



