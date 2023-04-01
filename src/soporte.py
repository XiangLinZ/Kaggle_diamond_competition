import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
from tqdm import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

def analisis_basico(dataframe):
    """
    Realiza un análisis básico de un dataframe y muestra información relevante sobre su estructura y variables.

    Args:
        dataframe (pandas.DataFrame): El dataframe que se desea analizar.

    Returns:
        None: La función no retorna ningún valor.
    """
    
    # Imprimir estructura del dataframe
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print(f"Estructura de los datos: {dataframe.shape}")
    display(dataframe.head(2))
    
    # Imprimir número de filas duplicadas
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Número de filas duplicadas:")
    print(dataframe.duplicated().sum())
    
    # Imprimir columnas, nulos y dtypes
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Columnas, Nulos y Dtypes:")
    display(pd.concat([dataframe.isnull().sum(), dataframe.dtypes], axis=1).rename(columns={0: "nulos", 1: "dtypes"}))
    
    # Imprimir descripción de las variables numéricas
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Descripción de las variables tipo Numéricas:")
    display(dataframe.describe().T)
    
    # Imprimir descripción de las variables categóricas
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Descripción de las variables tipo Categóricas:")
    display(dataframe.describe(include="object").T)
    

def distribucion_numericas(dataframe):
    """
    Genera un conjunto de gráficos de distribución (KDE) para las variables numéricas de un dataframe.

    Args:
        dataframe (pandas.DataFrame): El dataframe que se desea analizar.

    Returns:
        None: La función no retorna ningún valor.
    """
    # Obtener las columnas numéricas del dataframe
    columnas_numeric = dataframe.select_dtypes(include=np.number).columns

    # Crear el conjunto de subplots para graficar las distribuciones
    fig, axes = plt.subplots(nrows=int(np.ceil(len(columnas_numeric)/2)), ncols=2, figsize=(25, 15))
    axes = axes.flat

    # Crear una lista de colores aleatorios para cada variable
    lista_colores = []
    n = len(columnas_numeric)

    for i in range(n):
        lista_colores.append('#%03X' % random.randint(0, 0xFFF))

    # Graficar la distribución de cada variable
    for i, colum in enumerate(columnas_numeric):
        sns.kdeplot(
            data=dataframe,
            x=colum,
            color=lista_colores[i],
            shade=True,
            alpha=0.2,
            ax=axes[i])

        axes[i].set_title(colum, fontsize=15, fontweight="bold")
        axes[i].tick_params(labelsize=20)
        axes[i].set_xlabel("")

    fig.tight_layout()


def correla_respuesta_num(dataframe, respuesta):
    """Función que grafica la relación entre cada variable numérica de un dataframe y una variable de respuesta dada.

    Args:
        dataframe (pandas.DataFrame): Dataframe con las variables.
        respuesta (str): Nombre de la variable de respuesta.

    Returns:
        None.
    """
    columnas_numeric = dataframe.select_dtypes(include = np.number).columns
    columnas_numeric = columnas_numeric.drop(respuesta)

    fig, axes = plt.subplots(nrows = int(np.ceil(len(columnas_numeric)/2)), ncols = 2, figsize = (25, 15))
    axes = axes.flat

    lista_colores = []
    n = len(columnas_numeric)

    for i in range(n):
        lista_colores.append('#%03X' % random.randint(0, 0xFFF))

    for i, colum in enumerate(columnas_numeric):
        sns.regplot(
            x = dataframe[colum], 
            y = dataframe[respuesta], 
            color = lista_colores[i], 
            marker = ".", 
            scatter_kws = {"alpha": 0.4}, 
            line_kws = {"color": "red", "alpha": 0.7 }, 
            ax = axes[i])
        
        axes[i].set_title(f"{respuesta} vs {colum}", fontsize = 10, fontweight = "bold")
        axes[i].tick_params(labelsize = 20)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        
    fig.tight_layout();


def correla_map(dataframe):
    """
    Muestra un heatmap de correlación a partir de la matriz de correlación de un dataframe.

    Args:
        dataframe (pandas.DataFrame): El dataframe del que se quiere mostrar la matriz de correlación.

    Returns:
        None. Muestra el gráfico en pantalla.
    """
    mask = np.triu(np.ones_like(dataframe.corr(), dtype = bool))

    plt.figure(figsize = (15, 10))

    sns.heatmap(dataframe.corr(), 
        cmap = "YlGnBu", 
        mask = mask,
        annot = True);


def correla_respuesta_cate(dataframe, respuesta):
    """
    Función que permite visualizar la relación entre variables categóricas y la variable respuesta mediante gráficos de caja.
    
    Args:
        dataframe (pandas.DataFrame): Dataframe con las variables a analizar.
        respuesta (str): Nombre de la variable respuesta.
    
    Returns:
        None
    """
    # Seleccionar columnas categóricas
    columnas_object = dataframe.select_dtypes(include = "object").columns

    # Crear subplots para graficar
    fig, axes = plt.subplots(nrows = int(np.ceil(len(columnas_object)/2)), ncols = 2, figsize = (25, 15))
    axes = axes.flat

    # Iterar sobre cada columna categórica y graficar
    for i, colum in enumerate(columnas_object): 
        sns.boxplot(
            data = dataframe,
            x = colum,
            y = respuesta,
            ax = axes[i])
                
        axes[i].set_title(colum, fontsize = 15, fontweight = "bold")
        axes[i].tick_params(labelsize = 20)
        axes[i].set_xlabel("")

    fig.tight_layout();


def outlier_boxplot(dataframe, respuesta = None):
    """
    Función que genera un boxplot para cada columna numérica en un DataFrame,
    para identificar la presencia de valores atípicos (outliers).

    Args:
        dataframe: DataFrame. El DataFrame sobre el cual se desea generar los boxplots.
        respuesta: str (default None). El nombre de la columna que se considera como variable respuesta.

    Returns:
        None
    """
    columnas_numeric = dataframe.select_dtypes(include = np.number).columns

    if respuesta != None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    fig, axes = plt.subplots(len(columnas_numeric), 1, figsize = (25, 15))

    for i, colum in enumerate(columnas_numeric):
        sns.boxplot(
            x = columnas_numeric[i],
            data = dataframe,
            ax = axes[i],
            color = "royalblue")
        axes[i].set_title(colum, fontsize = 15, fontweight = "bold")
        axes[i].set_xlabel("")
    fig.tight_layout();


def detectar_outliers(dataframe, respuesta=None, diccionario={}):
    """
    Función que detecta los outliers de un dataframe numérico y los almacena en un diccionario.

    Parámetros:
    - dataframe: DataFrame. Dataframe del cual se quieren detectar los outliers.
    - respuesta: str, opcional. Nombre de la columna que se quiere utilizar como respuesta en un análisis de regresión. Default es None.
    - diccionario: dict, opcional. Diccionario que puede ser utilizado para especificar límites personalizados de detección de outliers para cada columna. Default es un diccionario vacío {}.

    Returns:
    - dicc_indices: dict. Diccionario que contiene las columnas y sus correspondientes índices de outliers.

    Ejemplo de uso:
    >> diccionario_outliers = detectar_outliers(dataframe, respuesta="precio", diccionario={"altura": {"bot": 130, "top": 200}})
    """
    dicc_indices = {} # creamos un diccionario donde almacenaremos índices de los outliers

    columnas_numeric = dataframe.select_dtypes(include = np.number).columns

    if respuesta != None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    # iteramos por la lista de las columnas numéricas de nuestro dataframe
    for col in tqdm(columnas_numeric):
        # calculamos los cuartiles Q1 y Q3
        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)
        
        # calculamos el rango intercuartil
        IQR = Q3 - Q1
        
        # calculamos los límites
        outlier_step = 1.5 * IQR

        if col in diccionario:
            # Checkeamos el límite por debajo
            if "bot" in diccionario[col]:
                outlier_step_bot = diccionario[col]["bot"]
            elif "bot" not in diccionario[col]:
                outlier_step_bot = Q1 - outlier_step
            
             # Checkeamos el límite por arroba
            if "top" in diccionario[col]:
                outlier_step_top = diccionario[col]["top"]
            elif "top" not in diccionario[col]:
                outlier_step_top = Q3 - outlier_step

        else:
            # Calculamos por donde cortar
            outlier_step_bot = Q1 - outlier_step
            outlier_step_top = Q3 + outlier_step

        # Filtramos nuestro dataframe para indentificar los outliers
        outliers_data = dataframe[(dataframe[col] < outlier_step_bot) | (dataframe[col] > outlier_step_top)]
        
        if outliers_data.shape[0] > 0: # chequeamos si nuestro dataframe tiene alguna fila. 
        
            dicc_indices[col] = (list(outliers_data.index)) # si tiene fila es que hay outliers y por lo tanto lo añadimos a nuestro diccionario
        
    return dicc_indices 

def tratar_outliers(dataframe, dic_outliers, metodo = "drop", value = 0):
    """
    Función que trata los outliers de un dataframe de acuerdo al método especificado.
    
    Args:
    - dataframe: dataframe que contiene los outliers a tratar.
    - dic_outliers (dict): diccionario que contiene los índices de los outliers a tratar.
    - metodo (str): método para tratar los outliers. Opciones disponibles: "drop", "mean", "median", "replace", "null".
    - value (float): valor que se utilizará en caso de elegir los métodos "mean", "median" o "replace".
    
    Returns:
    - dataframe2 (DataFrame): dataframe tratado de acuerdo al método especificado.
    """
    dataframe2 = dataframe.copy()

    if metodo == "drop":
        valores = set(sum((list(dic_outliers.values())), []))
        dataframe2 = dataframe.drop(dataframe.index[list(valores)])
    
    elif metodo in ["mean", "median", "replace", "null"]:
        for k, v in tqdm(dic_outliers.items()):
            if metodo == "mean":
                value = dataframe[k].mean() 
            
            elif metodo == "median":
                value = dataframe[k].median() 

            elif metodo == "null":
                value = np.nan

            else:
                pass
            
            for i in v: 
                dataframe2.loc[i,k] = value

    return dataframe2


def tratamiento_nulos_num(dataframe, metodo, valor = 0 , respuesta = None, neighbors = 5):
    """
    Función que realiza el tratamiento de valores nulos en columnas numéricas de un dataframe.
    
    Parámetros:
    - dataframe: dataframe que se desea transformar.
    - metodo: método para el tratamiento de valores nulos. Puede ser "drop", "replace", "mean", "median", "mode",
              "iterative" o "knn".
    - valor: valor para reemplazar los valores nulos en caso de elegir el método "replace".
    - respuesta: nombre de la columna respuesta, en caso de haberla, para no incluir en el tratamiento de valores nulos.
    - neighbors: número de vecinos a tener en cuenta en caso de elegir el método "knn".
    
    Retorna:
    - dataframe2: dataframe transformado.
    """
    # Obtenemos las columnas numéricas del dataframe
    columnas_numeric = dataframe.select_dtypes(include=np.number).columns

    if respuesta is not None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    # Método "drop": eliminamos las filas que contienen valores nulos
    if metodo == "drop":
        dataframe2 = dataframe[columnas_numeric].dropna(how="any")
        return dataframe2
    
    # Métodos "replace", "mean", "median", "mode": reemplazamos los valores nulos por otro valor
    elif metodo in ["replace", "mean", "median", "mode"]:
        if metodo == "replace":
            numericas_trans = dataframe[columnas_numeric].fillna(valor)

        else:
            for col in columnas_numeric:
                if metodo == "mean":
                    dataframe2 = dataframe[col].fillna(dataframe[col].mean()[0])
                elif metodo == "median":
                    dataframe2 = dataframe[col].fillna(dataframe[col].median()[0])
                elif metodo == "mode":
                    dataframe2 = dataframe[col].fillna(dataframe[col].mode()[0])
            return dataframe2 
        
    # Métodos "iterative" y "knn": utilizamos imputación para reemplazar los valores nulos
    elif metodo in ["iterative", "knn"]:
        if metodo == "iterative":
            imputer = IterativeImputer()
        elif metodo == "knn":
            imputer = KNNImputer(neighbors)

        numericas_trans = pd.DataFrame(imputer.fit_transform(dataframe[columnas_numeric]), columns=columnas_numeric)
    
    # Combinamos las columnas numéricas transformadas con el resto del dataframe
    dataframe2 = dataframe.drop(columnas_numeric, axis=1)
    dataframe2[columnas_numeric] = numericas_trans

    return dataframe2


def tratamiento_nulos_cat(dataframe, metodo = "drop", valor = "desconocido", respuesta = None):
    """
    Esta función se encarga de tratar los valores nulos de las columnas categóricas de un dataframe.

    Parameters:
    -----------
    dataframe: pandas.DataFrame
        El dataframe que se desea transformar.
    metodo: str, default "drop"
        El método de transformación de los valores nulos, las opciones son:
        - "drop": elimina las filas que contienen valores nulos.
        - "replace": reemplaza los valores nulos por un valor dado.
    valor: str, default "desconocido"
        El valor a utilizar para reemplazar los valores nulos si el método seleccionado es "replace".
    respuesta: list, default None
        Una lista con el nombre de las columnas a las cuales no se les aplicará el tratamiento.

    Returns:
    --------
    pandas.DataFrame
        El dataframe con los valores nulos de las columnas categóricas transformados según el método elegido.
    """
    columnas_object = dataframe.select_dtypes(include = "object").columns

    if respuesta != None:
        columnas_object = columnas_object.drop(respuesta)
        
    if metodo == "drop":
        dataframe2 = dataframe[columnas_object].dropna(how = "any")

    elif metodo == "replace":
        categoricas_trans = dataframe[columnas_object].fillna(valor)
        dataframe2 = dataframe.drop(columnas_object, axis = 1)
        dataframe2[columnas_object] = categoricas_trans

    return dataframe2


def encoder(dataframe, diccionario, modelo = 0):
    """
    Función que recibe un dataframe y un diccionario que contiene información
    sobre la codificación de las variables. Devuelve el dataframe codificado y guarda
    los objetos de codificación en archivos pickle.

    Parameters
    ----------
    dataframe : pandas DataFrame
        El dataframe que se desea codificar.
    diccionario : diccionario
        Diccionario que contiene la información sobre la codificación de las variables.
        Cada llave es el nombre de la variable y el valor es otro diccionario con información
        específica sobre la codificación. Si el valor es 'dummies' o 'one_hot', la variable
        se codificará mediante OneHotEncoder. Si el valor es 'label', se codificará mediante
        LabelEncoder. Si el valor es un diccionario, se usará información específica del
        diccionario para la codificación.
        - Ejemplo de uso:
        diccionario = {"Color": "one_hot",
                        "Región": "dummies",
                        "Tamaño": {"ordinal": ["pequeño", "mediano", "grande"]},
                        "Forma": "label",
                        "Edad": {"map": {"18": 1, "19": 1, "20": 2}}}
    modelo : int, optional
        El número del modelo, por defecto 0.

    Returns
    -------
    pandas DataFrame
        El dataframe codificado.
    """
    dataframe2 = dataframe.copy()

    for k, v in tqdm(diccionario.items()):
        if v in ["dummies", "one_hot"]:
            encoder = OneHotEncoder()
            dataframe2[k] = encoder.fit_transform(dataframe[k])

        elif v == "label":
            encoder = LabelEncoder()
            dataframe2[k] = encoder.fit_transform(dataframe[k])

        elif type(v) == dict:
            if list(v.keys())[0] == "ordinal":
                encoder = OrdinalEncoder(categories = [v["ordinal"]], dtype = int)
                dataframe2[k] = encoder.fit_transform(dataframe[[k]])

            elif list(v.keys())[0] == "map":
                encoder = diccionario[k]["map"]
                dataframe2[k] = dataframe[k].map(v["map"])

        with open(f'../data/encoding_{k}_{modelo}.pkl', 'wb') as s:
                pickle.dump(encoder, s)

    return dataframe2


def estandarizacion(dataframe, metodo, lista = None ,respuesta = None, modelo = 0):
    """
    Realiza la estandarización de las columnas numéricas de un dataframe utilizando el método "estandar" o "robust".

    Parameters:
    -----------
    dataframe : pandas DataFrame
        Dataframe a estandarizar.
    metodo : str
        Método de estandarización a utilizar. Debe ser "estandar" o "robust".
    lista : list or None, optional (default=None)
        Lista de columnas numéricas a estandarizar. Si no se especifica, se estandarizarán todas las columnas numéricas del dataframe.
    respuesta : str or None, optional (default=None)
        Nombre de la columna que se utilizará como respuesta en un modelo. Esta columna no se estandarizará.
    modelo : int, optional (default=0)
        Número de modelo para identificar los archivos generados por la función.

    Returns:
    --------
    pandas DataFrame
        Dataframe estandarizado.
    """
    dataframe2 = dataframe.copy()

    columnas_numeric = dataframe.select_dtypes(include = np.number).columns

    if respuesta != None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    if lista != None:
        columnas_numeric = lista

    if metodo == "estandar":
        scaler = StandardScaler()
    elif metodo == "robust":
        scaler = RobustScaler()

    dataframe2[columnas_numeric] = scaler.fit_transform(dataframe[columnas_numeric])

    with open(f'../data/estandarizacion_{modelo}.pkl', 'wb') as s:
        pickle.dump(scaler, s)

    return dataframe2


def mejores_parametros_num(dataframe, respuesta, random_state = 42, test_size = 0.2):
    """
    Encuentra los mejores parámetros para un modelo de árbol de decisión de regresión numérica.

    Parámetros:
    dataframe: DataFrame de pandas que contiene los datos de entrenamiento y prueba.
    respuesta: Nombre de la columna que contiene la variable respuesta.
    random_state: Semilla aleatoria para asegurar la reproducibilidad de los resultados (predeterminado = 42).
    test_size: Porcentaje de datos que se deben utilizar para el conjunto de prueba (predeterminado = 0.2).

    Retorna:
    Un diccionario con los mejores parámetros para el modelo de árbol de decisión.
    """
    # Dividir el conjunto de datos en entrenamiento y prueba
    X = dataframe.drop(respuesta, axis=1)
    y = dataframe[respuesta]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Crear un modelo de árbol de decisión y ajustarlo al conjunto de entrenamiento
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    # Definir los posibles valores de los parámetros
    max_features = np.sqrt(len(X_train.columns))
    max_depth = regressor.tree_.max_depth

    if max_depth > 15:
        lista_depth = [x for x in range(2, (15), 2)] + [x for x in range(16, (max_depth + 2), 4)]
    elif max_depth > 8:
        lista_depth = [x for x in range(2, (max_depth + 2), 2)]
    else:
        lista_depth = [x for x in range(2, (max_depth + 2))]
    
    param = {
        "max_depth": lista_depth,
        "min_samples_split": [x for x in range(25, 201, 25)],
        "min_samples_leaf": [x for x in range(25, 201, 25)],
        "max_features": [x for x in range(1, int(max_features + 2))]}
    
    # Buscar los mejores parámetros utilizando una búsqueda por cuadrícula
    grid_search = GridSearchCV(estimator=regressor, param_grid=param, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


def modelos_num(dataframe, respuesta, lista, parametros_tree = None, comparativa = True, modelo = 0, random_state = 42, test_size = 0.2, scoring = None):
    """
    Entrena y evalúa modelos de regresión numérica. Los modelos a entrenar pueden ser árboles de decisión, bosques aleatorios, 
    boosting de gradientes o k-NN. La función utiliza GridSearchCV para encontrar los mejores parámetros para cada modelo.

    Parámetros:
    -----------
    - dataframe : pandas.DataFrame
        El DataFrame que contiene los datos.
    - respuesta : str
        El nombre de la columna que contiene la variable respuesta.
    - lista : list
        La lista de modelos a entrenar. Puede ser una combinación de "tree", "forest", "gradient" o "knn".
    - parametros_tree : dict, default=None
        Los parámetros que se usarán en el modelo de árbol de decisión, bosque aleatorio y boosting de gradientes. Si no se proporcionan, 
        se usan los parámetros por defecto de scikit-learn.
    - comparativa : bool, default=True
        Si es verdadero, el modelo se evalúa en un conjunto de prueba. Si es falso, el modelo se entrena en todos los datos.
    - modelo : int, default=0
        Un número que se usa para identificar el modelo. Se utiliza para guardar los modelos y las métricas en archivos con nombres 
        únicos.
    - random_state : int, default=42
        La semilla aleatoria para dividir los datos en conjuntos de entrenamiento y prueba.
    - test_size : float, default=0.2
        El tamaño del conjunto de prueba.
    - scoring : str or callable, default=None
        La métrica para optimizar en el GridSearchCV. Si no se proporciona, se utiliza la métrica predeterminada de cada modelo.

    Retorna:
    --------
    - Si comparativa es verdadero, retorna un DataFrame con las métricas de evaluación del modelo para los conjuntos de entrenamiento 
      y prueba. Si comparativa es falso, no se retorna nada.

    Ejemplos:
    ---------
    # Entrenar un modelo de árbol de decisión y un bosque aleatorio con los parámetros predeterminados.
    modelos_num(df, 'target', ['tree', 'forest'], modelo=1)

    # Entrenar un modelo de k-NN con una métrica de distancia de Manhattan y el parámetro de vecinos de 5 y 10.
    parametros_knn = {'n_neighbors': [5, 10], 'metric': ['manhattan']}
    modelos_num(df, 'target', ['knn'], parametros_tree=parametros_knn, modelo=2)
    """
    X = dataframe.drop(respuesta, axis = 1)
    y = dataframe[respuesta]

    if comparativa == True:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size, random_state = random_state)
        df_metricas = pd.DataFrame({"MAE": [] ,"MSE":[], "RMSE":[],"R2":[], "set": [], "modelo":[]})
    else:
        X_train = X
        y_train = y

    for tipo_modelo in tqdm(lista):
        if tipo_modelo == "tree":
            estimador = DecisionTreeRegressor()
            parametros_search = parametros_tree
            tipo = "Decision_Tree"
        elif tipo_modelo == "forest":
            estimador = RandomForestRegressor()
            parametros_search = parametros_tree
            tipo = "Random_Forest"
        elif tipo_modelo == "gradient":
            estimador = GradientBoostingRegressor()
            parametros_search = parametros_tree
            tipo = "Gradient_Booster"
        elif tipo_modelo == "knn":
            estimador = KNeighborsRegressor()
            k_range = list(range(1, 31))
            parametros_search = dict(n_neighbors = k_range)
            tipo = f"KNN"

        gs = GridSearchCV(
            estimator = estimador,
            param_grid = parametros_search,
            cv = 10,
            verbose = 0,
            n_jobs = -1,
            return_train_score = True,
            scoring = scoring)
        gs.fit(X_train, y_train)

        modelo_final = gs

        if comparativa == True:
            y_pred_test = modelo_final.predict(X_test)
            y_pred_train = modelo_final.predict(X_train)
            resultados = {'MAE': [metrics.mean_absolute_error(y_test, y_pred_test), metrics.mean_absolute_error(y_train, y_pred_train)],
                        'MSE': [metrics.mean_squared_error(y_test, y_pred_test), metrics.mean_squared_error(y_train, y_pred_train)],
                        'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)), np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))],
                        'R2':  [metrics.r2_score(y_test, y_pred_test), metrics.r2_score(y_train, y_pred_train)],
                        "set": ["test", "train"]}
            dt_results = pd.DataFrame(resultados)
            dt_results["modelo"] = f"{tipo} {modelo}"
            df_metricas = pd.concat([df_metricas, dt_results], axis = 0)
            with open(f'../data/metricas_{modelo}.pkl', 'wb') as metric:
                pickle.dump(df_metricas, metric)

        with open(f'../data/modelo_{tipo}_v{modelo}.pkl', 'wb') as model:
                pickle.dump(modelo_final, model)
                
        with open(f'../data/best_parametros_{tipo}_v{modelo}.pkl', 'wb') as parametros:
            pickle.dump(modelo_final.best_params_, parametros)

    if comparativa == True:
        return df_metricas