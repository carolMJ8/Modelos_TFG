Esta carpeta contiene, además de los archivos donde se crean y entrenan los modelos explicados a continuación, el fichero *weights.h5* donde se incluyen los pesos del modelo que se utilizó
finalmente para la aplicación móvil (https://github.com/carolMJ8/TFG_App_Cliente y https://github.com/carolMJ8/TFG_App_Servidor), siendo este el que mejores resultados aportó
después de todos los experimentos realizados. A continuación se realizará una breve explicación del contenido de cada archivo. Como los ficheros *DataFunctions.py* y *DataFunctionsNoPres.py*
solo difieren en que el contenido del segundo no incluye los datos de la presión del dataset original, se explicarán las funciones de ambos en general y no individualmente por cada fichero.


ARCHIVOS DATAFUNCTIONS Y DATAFUNCTIONSNOPRES
- **get_data(data, tags)**
  - *Argumentos:* las rutas hacia los ficheros de datos y de etiquetas.
  - Se guarda el contenido de ambos archivos y se convierte al tipo de dato DataFrame. Una vez hecho esto se normalizan los datos con la transformación Z-score y se 
    eliminan de ambos ficheros las filas de datos cuyo etiquetado sea 0.
  - *Return:* se devuelve un dataframe con los datos y otro con las etiquetas.
 
- **get_arrays_from_data(data, tags)**
  - *Argumentos:* los dataframes de datos y etiquetas.
  - Se convierten cada una de las columnas de los datos de cada dataframe en arrays de NumPy. Después se almacenan todos ellos en un nuevo dataframe y en un diccionario.
  - *Return:* dataframe y diccionario con los arrays de NumPy.
 
- **divide_data_arrays_into_windows(array_dict, labels, moda, window_size)**
  - *Argumentos:* los arrays de NumPy con los datos, las etiquetas, un valor booleano indicando si utilizar la moda o no para el método de etiquetado de ventanas y el tamaño deseado de las ventanas.
  - Se guardan individualmente cada uno de los arrays de datos. También se almacena en otro array sus etiquetas. En un bucle, se itera sobre cada uno de los arrays de datos, aumentando el valor del
  	índice de iteración en un tamaño de window_size para crear las ventanas deslizantes para cada uno de los tipos de datos. Después se crea el etiquetado de las ventanas de acuerdo al argumento moda.
    Si tiene valor falso, la etiqueta de la ventana se corresponde con la etiqueta orginal del último elemento de la ventana. Si es verdadero, se utiliza como etiqueta general la resultante de calcular
    la moda de todas las etiquetas de todos los datos de cada uno de los datos que conforman la ventana deslizante. Al terminar este proceso, se guardan los arrays de ventanas deslizantes en un diccionario.
  - *Return:* diccionario con los arrays de ventanas deslizantes y el array de etiquetas de las ventanas.
 
- **divide_train_test(x, y, train_percent)**
  - *Argumentos:* diccionario con los datos, array de etiquetas y el porcentaje con el que queremos dividir el conjunto de datos.
  - Primero se calcula el tamaño del subconjunto de entrenamiento con el argumento train_percent. Seguidamente se divide cada uno de los arrays de datos según el valor del tamaño calculado para el
    subconjunto de entrenamiento y el de pruebas. Finalmente se guardan en dos diccionarios los arrays de los datos de cada subconjunto.
  - *Return:* los diccionarios de datos y los arrays de etiquetas de entrenamiento y de pruebas.
 
- **get_data_and_create_windows(data_path, labels_paths, less_data=False, num_data=0, title=None)**
  - *Argumentos:* rutas hacia los archivos de datos y etiquetas, valor booleano indicando si seleccionar una menor porción de los datos y la cantidad de los mismos (less_data, num_data usados para
    pruebas) y nombre del archivo de datos que estamos tratando.
  - Se agrupa en un solo método toda la funcionalidad descrita anteriormente. Además, se imprime por pantalla un texto con información sobre los datos: nombre del archivo, tamaño, número y 
    forma de las ventanas de entrenamiento y pruebas.

- **create_full_train_test_arrays(nuevo=False, x_train_user1_220617_hips=None, x_test_user1_220617_hips=None, ... , y_train_user1_220617=None, y_test_user1_220617=None, ... )**
