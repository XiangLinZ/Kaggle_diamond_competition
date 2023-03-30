import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OrdinalEncoder

import warnings
warnings.filterwarnings('ignore')

def analisis_basico(dataframe):
    """_summary_

    Args:
        dataframe (_type_): _description_
    """
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print (f"Estructura de los datos: {dataframe.shape}")
    display(dataframe.head(2))
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Número de filas duplicadas:") 
    print(dataframe.duplicated().sum())
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Columnas, Nulos y Dtypes:") 
    display(pd.concat([dataframe.isnull().sum(), dataframe.dtypes], axis = 1).rename(columns = {0: "nulos", 1: "dtypes"}))
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Descripción de las variables tipo Numéricas:")
    display(dataframe.describe().T)
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    print("Descripción de las variables tipo Categóricas:")
    display(dataframe.describe(include = "object").T)
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n")
    return None

def distribucion_numericas(dataframe):
    columnas_numeric = dataframe.select_dtypes(include = np.number).columns

    fig, axes = plt.subplots(nrows = int(np.ceil(len(columnas_numeric)/2)), ncols = 2, figsize = (25, 15))
    axes = axes.flat

    lista_colores = []
    n = len(columnas_numeric)
    for i in range(n):
        lista_colores.append('#%03X' % random.randint(0, 0xFFF))

    for i, colum in enumerate(columnas_numeric): 
        sns.kdeplot( 
            data = dataframe,
            x = colum,
            color = lista_colores[i], 
            shade = True, 
            alpha = 0.2, 
            ax = axes[i])
        
        axes[i].set_title(colum, fontsize = 15, fontweight = "bold")
        axes[i].tick_params(labelsize = 20)
        axes[i].set_xlabel("")
        
    fig.tight_layout();  
    return None


def correla_respuesta_num(dataframe, respuesta):
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
    return None

def correla_map(dataframe):
    mask = np.triu(np.ones_like(dataframe.corr(), dtype = bool))
    plt.figure(figsize = (15, 10))
    sns.heatmap(dataframe.corr(), 
        cmap = "YlGnBu", 
        mask = mask,
        annot = True);
    return None

def correla_respuesta_cate(dataframe, respuesta):
    columnas_object = dataframe.select_dtypes(include = "object").columns

    fig, axes = plt.subplots(nrows = int(np.ceil(len(columnas_object)/2)), ncols = 2, figsize = (25, 15))
    axes = axes.flat

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
    return None

def outlier_boxplot(dataframe, respuesta = None):
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
    return None

def detectar_outliers(dataframe, respuesta = None, diccionario = {}): 
    
    dicc_indices = {} # creamos un diccionario donde almacenaremos índices de los outliers
    columnas_numeric = dataframe.select_dtypes(include = np.number).columns
    if respuesta != None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    # iteramos por la lista de las columnas numéricas de nuestro dataframe
    for col in columnas_numeric:
                #calculamos los cuartiles Q1 y Q3
        Q1 = np.nanpercentile(dataframe[col], 25)
        Q3 = np.nanpercentile(dataframe[col], 75)
        
        # calculamos el rango intercuartil
        IQR = Q3 - Q1
        
        # calculamos los límites
        outlier_step = 1.5 * IQR

        if col in diccionario:
            if "bot" in diccionario[col]:
                outlier_step_bot = diccionario[col]["bot"]
            elif "bot" not in diccionario[col]:
                outlier_step_bot = Q1 - outlier_step
            
            if "top" in diccionario[col]:
                outlier_step_top = diccionario[col]["top"]
            elif "top" not in diccionario[col]:
                outlier_step_top = Q3 - outlier_step

        else:
            outlier_step_bot = Q1 - outlier_step
            outlier_step_top = Q3 + outlier_step
            # filtramos nuestro dataframe para indentificar los outliers
        
        outliers_data = dataframe[(dataframe[col] < outlier_step_bot) | (dataframe[col] > outlier_step_top)]
        
        
        if outliers_data.shape[0] > 0: # chequeamos si nuestro dataframe tiene alguna fila. 
        
            dicc_indices[col] = (list(outliers_data.index)) # si tiene fila es que hay outliers y por lo tanto lo añadimos a nuestro diccionario
        
    return dicc_indices 

def tratar_outliers(dataframe, dic_outliers, metodo = "drop", value = 0):
    dataframe2 = dataframe.copy()
    if metodo == "drop":
        valores = set(sum((list(dic_outliers.values())), []))
        dataframe2 = dataframe.drop(dataframe.index[list(valores)])
    
    elif metodo in ["mean", "median", "replace", "null"]:
        for k, v in dic_outliers.items():
            if metodo == "mean":
                value = dataframe[k].mean() # calculamos la media para cada una de las columnas que tenemos en nuestro diccionario
            
            elif metodo == "median":
                value = dataframe[k].median() # calculamos la mediana para cada una de las columnas que tenemos en nuestro diccionario

            elif metodo == "null":
                value = np.nan

            else:
                pass
            
            for i in v: # iteremos por la lista de valores para cada columna
                dataframe2.loc[i,k] = value

    return dataframe2


def tratamiento_nulos_num(dataframe, metodo, valor = 0 , respuesta = None, neighbors = 5):
    columnas_numeric = dataframe.select_dtypes(include = np.number).columns
    if respuesta != None:
        columnas_numeric = columnas_numeric.drop(respuesta)

    if metodo == "drop":
        dataframe2 = dataframe[columnas_numeric].dropna(how = "any")
        return dataframe2
    
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
        
    elif metodo in ["iterative", "knn"]:
        if metodo == "iterative":
            imputer = IterativeImputer()
        elif metodo == "knn":
            imputer = KNNImputer(neighbors)

        numericas_trans = pd.DataFrame(imputer.fit_transform(dataframe[columnas_numeric]), columns = columnas_numeric)
    dataframe2 = dataframe.drop(columnas_numeric, axis = 1)
    dataframe2[columnas_numeric] = numericas_trans

    return dataframe2


def tratamiento_nulos_cat(dataframe, metodo = "drop", valor = "desconocido", respuesta = None):
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
    dataframe2 = dataframe.copy()
    for k, v in diccionario.items():
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
                dataframe2[k] = dataframe[k].map(v["map"])

        with open(f'../data/encoding_{k}_{modelo}.pkl', 'wb') as s:
                pickle.dump(encoder, s)

    return dataframe2

def estandarizacion(dataframe, metodo, lista = None ,respuesta = None, modelo = 0):
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