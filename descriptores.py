# descriptores.py

import numpy as np
import pandas as pd

def calcular_descriptores_segmento(segmento):
    """
    Calcula descriptores estadísticos por canal de EEG para un segmento dado.
    Estos descriptores resumen las características básicas de la señal.

    Parámetro:
    - segmento: arreglo NumPy de forma (canales, muestras)

    Devuelve:
    - DataFrame de pandas con los descriptores por canal:
      * Varianza
      * Desviación estándar
      * Promedio del valor absoluto
      * Autocorrelación (lag 1)
      * Autocovarianza (lag 1)
    """

    varianza = np.var(segmento, axis=1)
    std = np.std(segmento, axis=1)
    prom_abs = np.mean(np.abs(segmento), axis=1)

    # Autocorrelación: valor en lag 1
    autocorrelacion = [
        np.corrcoef(canal[:-1], canal[1:])[0, 1] if canal.std() > 0 else 0
        for canal in segmento
    ]

    # Autocovarianza (lag 1)
    autocovarianza = [
        np.cov(canal[:-1], canal[1:])[0, 1] if canal.std() > 0 else 0
        for canal in segmento
    ]

    # Empaquetar los resultados en un DataFrame para facilitar la visualización
    df = pd.DataFrame({
        "Varianza": varianza,
        "Desviación estándar": std,
        "Prom. valor absoluto": prom_abs,
        "Autocorrelación": autocorrelacion,
        "Autocovarianza": autocovarianza
    })

    return df

def calcular_matrices_descriptores(segmento):
    """
    Calcula matrices de relación entre canales para un segmento.

    Parámetro:
    - segmento: arreglo NumPy de forma (canales, muestras)

    Devuelve:
    - Diccionario con:
      * Matriz de covarianza
      * Matriz de correlación de Pearson
      * Matriz de correlación cruzada (lag 0)
    """
    # Covarianza entre canales
    covarianza = np.cov(segmento)

    # Pearson (correlación lineal entre canales)
    correlacion_pearson = np.corrcoef(segmento)

    # Correlación cruzada (lag 0)
    n = segmento.shape[1]
    cruzada = np.zeros((segmento.shape[0], segmento.shape[0]))
    for i in range(segmento.shape[0]):
        for j in range(segmento.shape[0]):
            cruzada[i, j] = np.correlate(segmento[i], segmento[j]) / n

    return {
        "Covarianza": covarianza,
        "Correlación de Pearson": correlacion_pearson,
        "Correlación cruzada (lag 0)": cruzada
    }

def calcular_todos_los_descriptores(segmentos):
    """
    Aplica todos los descriptores a cada segmento de la señal (before, seizure, after).

    Parámetro:
    - segmentos: diccionario con segmentos (clave: nombre, valor: matriz de señal)

    Devuelve:
    - Diccionario anidado con:
      * descriptores por canal ("canal_a_canal")
      * matrices de relación entre canales ("matrices")
    """
    resultados = {}
    for nombre, bloque in segmentos.items():
        resultados[nombre] = {
            "canal_a_canal": calcular_descriptores_segmento(bloque),
            "matrices": calcular_matrices_descriptores(bloque)
        }
    return resultados
