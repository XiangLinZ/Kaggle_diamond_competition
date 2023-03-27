import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

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


def correla_respuesta(dataframe, respuesta):
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