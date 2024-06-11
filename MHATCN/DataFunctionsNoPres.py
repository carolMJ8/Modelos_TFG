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

def get_data(data, tags):
    data = pd.read_csv(data).fillna(0)
    labels = pd.read_csv(tags, sep= ' ', names=['Tiempo', 'Etiqueta', 'Tipo Carretera'])
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data_norm.reset_index(drop=True, inplace=True)

    df = pd.concat([data_norm, labels[['Tiempo', 'Etiqueta']]], axis=1)
    df_reduced = df[df.iloc[:, -1] != 0]
    data_final = df_reduced[['Tiempo', 'Acelerometro en el eje X', 'Acelerometro en el eje Y', 'Acelerometro en el eje Z', 'Magnetometro en el eje X', 'Magnetometro en el eje Y', 'Magnetometro en el eje Z', 'Giroscopio en el eje X', 'Giroscopio en el eje Y', 'Giroscopio en el eje Z']]
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

    df = pd.concat([df_acel_x, df_acel_y, df_acel_z, df_magnetometro_x, df_magnetometro_y, df_magnetometro_z, df_giroscopio_x, df_giroscopio_y, df_giroscopio_z], axis=1)

    array_dict['AcelX'] = acel_x
    array_dict['AcelY'] = acel_y
    array_dict['AcelZ'] = acel_z
    array_dict['MagX'] = magnetometro_x
    array_dict['MagY'] = magnetometro_y
    array_dict['MagZ'] = magnetometro_z
    array_dict['GirosX'] = giroscopio_x
    array_dict['GirosY'] = giroscopio_y
    array_dict['GirosZ'] = giroscopio_z
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

    array_dict_test['AcelX'] = np.array(acel_x[tam_train:]).reshape(np.array(acel_x[tam_train:]).shape[0], np.array(acel_x[tam_train:]).shape[1], 1)
    array_dict_test['AcelY'] = np.array(acel_y[tam_train:]).reshape(np.array(acel_y[tam_train:]).shape[0], np.array(acel_y[tam_train:]).shape[1], 1)
    array_dict_test['AcelZ'] = np.array(acel_z[tam_train:]).reshape(np.array(acel_z[tam_train:]).shape[0], np.array(acel_z[tam_train:]).shape[1], 1)
    array_dict_test['MagX'] = np.array(magnetometro_x[tam_train:]).reshape(np.array(magnetometro_x[tam_train:]).shape[0], np.array(magnetometro_x[tam_train:]).shape[1], 1)
    array_dict_test['MagY'] = np.array(magnetometro_y[tam_train:]).reshape(np.array(magnetometro_y[tam_train:]).shape[0], np.array(magnetometro_y[tam_train:]).shape[1], 1)
    array_dict_test['MagZ'] = np.array(magnetometro_z[tam_train:]).reshape(np.array(magnetometro_z[tam_train:]).shape[0], np.array(magnetometro_z[tam_train:]).shape[1], 1)
    array_dict_test['GirosX'] = np.array(giroscopio_x[tam_train:]).reshape(np.array(giroscopio_x[tam_train:]).shape[0], np.array(giroscopio_x[tam_train:]).shape[1], 1)
    array_dict_test['GirosY'] = np.array(giroscopio_y[tam_train:]).reshape(np.array(giroscopio_y[tam_train:]).shape[0], np.array(giroscopio_y[tam_train:]).shape[1], 1)
    array_dict_test['GirosZ'] = np.array(giroscopio_z[tam_train:]).reshape(np.array(giroscopio_z[tam_train:]).shape[0], np.array(giroscopio_z[tam_train:]).shape[1], 1)

    return array_dict_train, np.array(y[:tam_train]), array_dict_test, np.array(y[tam_train:])

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

def create_full_train_test_arrays(nuevo=False, x_train_user1_220617_hips=None, x_test_user1_220617_hips=None,
                                   x_train_user1_260617_hips=None, x_test_user1_260617_hips=None,
                                   x_train_user2_180717_hand=None, x_test_user2_180717_hand=None,
                                   x_train_user3_140617_hips=None, x_test_user3_140617_hips=None,
                                   x_train_user3_070717_hips=None, x_test_user3_070717_hips=None,
                                   y_train_user1_220617=None, y_train_user1_260617=None, y_train_user2_180717=None, y_train_user3_140617=None, y_train_user3_070717=None,
                                   y_test_user1_220617=None, y_test_user1_260617=None, y_test_user2_180717=None, y_test_user3_140617=None, y_test_user3_070717=None):
    
    if nuevo is True:
        x_train, x_test, y_train, y_test = create_train_test_arrays_reduced(x_train_user1_260617_hips, x_train_user2_180717_hand, x_test_user1_260617_hips, x_test_user2_180717_hand, y_train_user1_260617, y_train_user2_180717, y_test_user1_260617, y_test_user2_180717)
        return x_train, x_test, y_train, y_test

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

    acelx_test = np.concatenate(acelx_test_array, axis=0)
    acely_test = np.concatenate(acely_test_array, axis=0)
    acelz_test = np.concatenate(acelz_test_array, axis=0)

    magx_test = np.concatenate(magx_test_array, axis=0)
    magy_test = np.concatenate(magy_test_array, axis=0)
    magz_test = np.concatenate(magz_test_array, axis=0)

    girosx_test = np.concatenate(girosx_test_array, axis=0)
    girosy_test = np.concatenate(girosy_test_array, axis=0)
    girosz_test = np.concatenate(girosz_test_array, axis=0)

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
    
    x_test['AcelX'] = acelx_test
    x_test['AcelY'] = acely_test
    x_test['AcelZ'] = acelz_test
    x_test['MagX'] = magx_test
    x_test['MagY'] = magy_test
    x_test['MagZ'] = magz_test
    x_test['GirosX'] = girosx_test
    x_test['GirosY'] = girosy_test
    x_test['GirosZ'] = girosz_test

    return x_train, x_test, y_train, y_test

def create_train_test_arrays_reduced(x_train_user1_260617_hips, x_train_user2_180717_hand, 
                                     x_test_user1_260617_hips, x_test_user2_180717_hand, 
                                     y_train_user1_260617, y_train_user2_180717, y_test_user1_260617, y_test_user2_180717):


    # ---------------- ACELEROMETRO EN X TRAIN
    acelx_train_user1_260617_hips = x_train_user1_260617_hips['AcelX']
    acelx_train_user2_180717_hand = x_train_user2_180717_hand['AcelX']

    # ---------------- ACELEROMETRO EN X TEST
    acelx_test_user1_260617_hips = x_test_user1_260617_hips['AcelX']
    acelx_test_user2_180717_hand = x_test_user2_180717_hand['AcelX']


    # ---------------- ACELEROMETRO EN Y TRAIN
    acely_train_user1_260617_hips = x_train_user1_260617_hips['AcelY']
    acely_train_user2_180717_hand = x_train_user2_180717_hand['AcelY']

    # ---------------- ACELEROMETRO EN Y TEST
    acely_test_user1_260617_hips = x_test_user1_260617_hips['AcelY']
    acely_test_user2_180717_hand = x_test_user2_180717_hand['AcelY']


    # ---------------- ACELEROMETRO EN Z TRAIN
    acelz_train_user1_260617_hips = x_train_user1_260617_hips['AcelZ']
    acelz_train_user2_180717_hand = x_train_user2_180717_hand['AcelZ']

    # ---------------- ACELEROMETRO EN Z TEST
    acelz_test_user1_260617_hips = x_test_user1_260617_hips['AcelZ']
    acelz_test_user2_180717_hand = x_test_user2_180717_hand['AcelZ']


    # ---------------- MAGNETOMETRO EN X TRAIN
    magx_train_user1_260617_hips = x_train_user1_260617_hips['MagX']
    magx_train_user2_180717_hand = x_train_user2_180717_hand['MagX']

    # ---------------- MAGNETOMETRO EN X TEST
    magx_test_user1_260617_hips = x_test_user1_260617_hips['MagX']
    magx_test_user2_180717_hand = x_test_user2_180717_hand['MagX']


    # ---------------- MAGNETOMETRO EN Y TRAIN
    magy_train_user1_260617_hips = x_train_user1_260617_hips['MagY']
    magy_train_user2_180717_hand = x_train_user2_180717_hand['MagY']

    # ---------------- MAGNETOMETRO EN Y TEST
    magy_test_user1_260617_hips = x_test_user1_260617_hips['MagY']
    magy_test_user2_180717_hand = x_test_user2_180717_hand['MagY']


    # ---------------- MAGNETOMETRO EN Z TRAIN
    magz_train_user1_260617_hips = x_train_user1_260617_hips['MagZ']
    magz_train_user2_180717_hand = x_train_user2_180717_hand['MagZ']

    # ---------------- MAGNETOMETRO EN Z TEST
    magz_test_user1_260617_hips = x_test_user1_260617_hips['MagZ']
    magz_test_user2_180717_hand = x_test_user2_180717_hand['MagZ']


    # ---------------- GIROSCOPIO EN X TRAIN
    girosx_train_user1_260617_hips = x_train_user1_260617_hips['GirosX']
    girosx_train_user2_180717_hand = x_train_user2_180717_hand['GirosX']

    # ---------------- GIROSCOPIO EN X TEST
    girosx_test_user1_260617_hips = x_test_user1_260617_hips['GirosX']
    girosx_test_user2_180717_hand = x_test_user2_180717_hand['GirosX']


    # ---------------- GIROSCOPIO EN Y TRAIN
    girosy_train_user1_260617_hips = x_train_user1_260617_hips['GirosY']
    girosy_train_user2_180717_hand = x_train_user2_180717_hand['GirosY']

    # ---------------- GIROSCOPIO EN Y TEST
    girosy_test_user1_260617_hips = x_test_user1_260617_hips['GirosY']
    girosy_test_user2_180717_hand = x_test_user2_180717_hand['GirosY']


    # ---------------- GIROSCOPIO EN Z TRAIN
    girosz_train_user1_260617_hips = x_train_user1_260617_hips['GirosZ']
    girosz_train_user2_180717_hand = x_train_user2_180717_hand['GirosZ']

    # ---------------- GIROSCOPIO EN Z TEST
    girosz_test_user1_260617_hips = x_test_user1_260617_hips['GirosZ']
    girosz_test_user2_180717_hand = x_test_user2_180717_hand['GirosZ']

    acelx_train_array = [acelx_train_user1_260617_hips, acelx_train_user2_180717_hand]

    acelx_test_array = [acelx_test_user1_260617_hips, acelx_test_user2_180717_hand]
    
    acely_train_array = [acely_train_user1_260617_hips, acely_train_user2_180717_hand]

    acely_test_array = [acely_test_user1_260617_hips, acely_test_user2_180717_hand]

    acelz_train_array = [acelz_train_user1_260617_hips, acelz_train_user2_180717_hand]

    acelz_test_array = [acelz_test_user1_260617_hips, acelz_test_user2_180717_hand]

    magx_train_array = [magx_train_user1_260617_hips, magx_train_user2_180717_hand]

    magx_test_array = [magx_test_user1_260617_hips, magx_test_user2_180717_hand]

    magy_train_array = [magy_train_user1_260617_hips, magy_train_user2_180717_hand]

    magy_test_array = [magy_test_user1_260617_hips, magy_test_user2_180717_hand]

    magz_train_array = [magz_train_user1_260617_hips, magz_train_user2_180717_hand]

    magz_test_array = [magz_test_user1_260617_hips, magz_test_user2_180717_hand]

    girosx_train_array = [girosx_train_user1_260617_hips, girosx_train_user2_180717_hand]

    girosx_test_array = [girosx_test_user1_260617_hips, girosx_test_user2_180717_hand]

    girosy_train_array = [girosy_train_user1_260617_hips, girosy_train_user2_180717_hand]

    girosy_test_array = [girosy_test_user1_260617_hips, girosy_test_user2_180717_hand]

    girosz_train_array = [girosz_train_user1_260617_hips, girosz_train_user2_180717_hand]

    girosz_test_array = [girosz_test_user1_260617_hips, girosz_test_user2_180717_hand]

    y_train_array = [y_train_user1_260617, y_train_user2_180717]

    y_test_array = [y_test_user1_260617, y_test_user2_180717]

    acelx_train = np.concatenate(acelx_train_array, axis=0)
    acely_train = np.concatenate(acely_train_array, axis=0)
    acelz_train = np.concatenate(acelz_train_array, axis=0)

    magx_train = np.concatenate(magx_train_array, axis=0)
    magy_train = np.concatenate(magy_train_array, axis=0)
    magz_train = np.concatenate(magz_train_array, axis=0)

    girosx_train = np.concatenate(girosx_train_array, axis=0)
    girosy_train = np.concatenate(girosy_train_array, axis=0)
    girosz_train = np.concatenate(girosz_train_array, axis=0)

    acelx_test = np.concatenate(acelx_test_array, axis=0)
    acely_test = np.concatenate(acely_test_array, axis=0)
    acelz_test = np.concatenate(acelz_test_array, axis=0)

    magx_test = np.concatenate(magx_test_array, axis=0)
    magy_test = np.concatenate(magy_test_array, axis=0)
    magz_test = np.concatenate(magz_test_array, axis=0)

    girosx_test = np.concatenate(girosx_test_array, axis=0)
    girosy_test = np.concatenate(girosy_test_array, axis=0)
    girosz_test = np.concatenate(girosz_test_array, axis=0)

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
    
    x_test['AcelX'] = acelx_test
    x_test['AcelY'] = acely_test
    x_test['AcelZ'] = acelz_test
    x_test['MagX'] = magx_test
    x_test['MagY'] = magy_test
    x_test['MagZ'] = magz_test
    x_test['GirosX'] = girosx_test
    x_test['GirosY'] = girosy_test
    x_test['GirosZ'] = girosz_test

    return x_train, x_test, y_train, y_test


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

    inputs = [input_acel_x, input_acel_y, input_acel_z, input_mag_x, input_mag_y, input_mag_z, input_giros_x, input_giros_y, input_giros_z]

    output_layer = Concatenate()([output_acel_x, output_acel_y, output_acel_z, output_mag_x, output_mag_y, output_mag_z, output_giros_x, output_giros_y, output_giros_z])

    return output_layer, inputs

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