# Modelos_TFG

En este repositorio se incluyen los archivos donde se han entrenado cada uno de los distintos modelos que se han creado para el Trabajo de Fin de Grado de Carolina Monedero Juzgado para la Universidad Autónoma de Madrid.
Cada arquitectura se encuentra dentro de su carpeta correspondiente donde se incluye al menos un archivo donde se encuentran las funciones de recopilación y preprocesamiento de datos además de las que se encargan de
construir cada uno de los modelos:

- **CNN**. Esta carpeta contiene el notebook *CNN_New* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar el modelo de **redes neuronales convolucionales**. El archivo *DataFunctions.py* contiene las funciones donde se realizan todos estos procedimientos.
- **LSTM**. Esta carpeta contiene el notebook **lstm* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar el modelo **Long-Short Term Memory**. El archivo *SlidingWindows.py* contiene las funciones donde se realizan todos estos procedimientos.
- **MHA**. Esta carpeta contiene el notebook *mha* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar el modelo de **atención multiple**. El archivo *DataFunctions.py* contiene las funciones donde se realizan todos estos procedimientos.
- **MHATCN**. Esta carpeta contiene los notebooks *mhatcn_all_data* y *mhatcn_full_dataset* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar los modelos **temporales convolucionales de atención múltiple**. Los archivos *DataFunctions.py* y *DataFunctionsNoPres.py* contienen las funciones donde se realizan todos estos procedimientos.
- **MLP**. Esta carpeta contiene el notebook *Perceptron_New* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar el modelo de **perceptrón multicapa**. El archivo *GetData.py* contiene las funciones donde se realizan todos estos procedimientos.
- **TCN**. Esta carpeta contiene el notebook *TCN_New* donde se llama a las funciones que recopilan y preprocesan los datos para después crear, entrenar y evaluar el modelo **temporal convolucional**. El archivo *DataFunctions.py* contiene las funciones donde se realizan todos estos procedimientos.

Por simplicidad, los archivos ".csv" y ".txt" donde se encuentran los datos originales con sus etiquetas, no han sido subidos al repositorio.
