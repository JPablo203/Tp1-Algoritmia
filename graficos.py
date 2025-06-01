# graficos.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np


def preparar_df_comparativo(resultados):
    """
    Une los resultados de cada segmento (before, seizure, after) en un solo DataFrame,
    agregando una columna 'bloque' que indica el origen de los datos.

    Parámetro:
    - resultados: diccionario con los DataFrames por bloque

    Retorna:
    - Un DataFrame con todos los datos combinados
    """
    bloques = []
    for nombre, data in resultados.items():
        df = data["canal_a_canal"].copy()
        df["bloque"] = nombre
        bloques.append(df)
    return pd.concat(bloques, ignore_index=True)

def graficar_histograma(df_completo, descriptor, carpeta):
    """
    Genera un histograma comparando la distribución de un descriptor
    entre los distintos bloques (before, seizure, after), con escala ajustada.
    """
    plt.figure(figsize=(10, 6))
    q1, q99 = np.percentile(df_completo[descriptor].dropna(), [1, 99])
    sns.histplot(data=df_completo, x=descriptor, hue="bloque", kde=True, element="step", stat="density")
    plt.xlim(q1, q99)
    plt.title(f"Histograma de {descriptor}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"histograma_{descriptor}.png"))
    plt.close()

def graficar_boxplot(df_completo, descriptor, carpeta):
    """
    Genera un boxplot (diagrama de cajas) del descriptor, separando por bloque.
    Útil para ver la dispersión y valores atípicos.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_completo, x="bloque", y=descriptor)
    plt.title(f"Diagrama de Cajas de {descriptor}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"boxplot_{descriptor}.png"))
    plt.close()

def graficar_scatterplot(df_completo, desc_x, desc_y, carpeta):
    """
    Grafica un gráfico de dispersión entre dos descriptores.
    Permite ver correlaciones o separaciones entre grupos.

    Parámetros:
    - desc_x: descriptor en el eje X
    - desc_y: descriptor en el eje Y
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_completo, x=desc_x, y=desc_y, hue="bloque")
    plt.title(f"Dispersión: {desc_x} vs {desc_y}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"scatter_{desc_x}_{desc_y}.png"))
    plt.close()

def graficar_heatmap_pearson(matriz, nombre_bloque, carpeta):
    """
    Genera un mapa de calor (heatmap) de la matriz de correlación de Pearson para un bloque.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, cmap="coolwarm", square=True, annot=True, fmt=".1f", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlación de Pearson - {nombre_bloque}")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f"heatmap_pearson_{nombre_bloque}.png"))
    plt.close()

def generar_todos_los_graficos(resultados, nombre_escenario="escenario1"):
    """
    Función principal que genera todas las gráficas para los descriptores:
    histogramas, diagramas de cajas y gráficos de dispersión.

    Parámetros:
    - resultados: salida de `calcular_todos_los_descriptores`
    - nombre_escenario: nombre de la carpeta donde se guardan las figuras
    """
    carpeta = os.path.join("figuras", nombre_escenario)
    os.makedirs(carpeta, exist_ok=True)

    df = preparar_df_comparativo(resultados)

    descriptores = ["Varianza", "Desviación estándar", "Prom. valor absoluto", "Autocorrelación", "Autocovarianza"]

    for descriptor in descriptores:
        graficar_histograma(df, descriptor, carpeta)
        graficar_boxplot(df, descriptor, carpeta)

    # Scatterplots sugeridos
    graficar_scatterplot(df, "Varianza", "Desviación estándar", carpeta)
    graficar_scatterplot(df, "Prom. valor absoluto", "Autocorrelación", carpeta)

    for nombre_bloque, data in resultados.items():
        matriz_pearson = data["matrices"]["Correlación de Pearson"]
        graficar_heatmap_pearson(matriz_pearson, nombre_bloque, carpeta)
