# main.py

#Se separo el funcionamiento del codigo en distintos archivos para que sea mas comodo
from procesamiento import cargar_datos, segmentar_senal
from descriptores import calcular_todos_los_descriptores
from graficos import generar_todos_los_graficos

#librerias usadas en main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Ubicacion de los archivos .edf a analizar (Tomar nota que se utilizo un chb para hacer un codigo plantilla)
archivo_edf = "datos/chb02/chb02_16.edf"
archivo_summary = "datos/chb02/chb02-summary.txt"

# Carga la señal EEG, frecuencia de muestreo y metadatos de crisis desde los archivos
senal, fs, canales, t_ini, t_fin = cargar_datos(archivo_edf, archivo_summary)

# Divide la señal en tres bloques: antes de la crisis (before), durante (seizure) y después (after)
segmentos = segmentar_senal(senal, fs, t_ini, t_fin)

# Calcula descriptores estadísticos para cada bloque y canal
resultados = calcular_todos_los_descriptores(segmentos)

# Información básica del archivo cargado
print("Canales:", canales)
print("Frecuencia de muestreo:", fs)
print("Segmentos extraídos:")
for nombre, bloque in segmentos.items():
    print(f" - {nombre}: {bloque.shape}")

# Escenario 2: calcula los descriptores sobre ventanas deslizantes de la señal centrada
senal_centrada = senal - np.mean(senal, axis=1, keepdims=True)
ventana_muestras = int(fs * 5)   # 5 segundos
paso_muestras = int(fs * 1)      # paso de 1 segundo

segmentos_ventaneados = []
for inicio in range(0, senal.shape[1] - ventana_muestras + 1, paso_muestras):
    ventana = senal_centrada[:, inicio:inicio + ventana_muestras]
    segmentos_ventaneados.append(ventana)

# Calcular descriptores por ventana y combinar
descriptores_completo = [calcular_todos_los_descriptores({"ventana": v})["ventana"]["canal_a_canal"]
                         for v in segmentos_ventaneados]

# Concatenar todos los descriptores canal a canal
df_completo = pd.concat(descriptores_completo, ignore_index=True)
resultados_totales = {
    "completo": {
        "canal_a_canal": df_completo,
        "matrices": calcular_todos_los_descriptores({"ventana": senal_centrada})["ventana"]["matrices"]
    }
}

# Escenario 1: genera los gráficos por bloque (before, seizure, after)  
generar_todos_los_graficos(resultados, nombre_escenario="escenario1")

# Escenario 2: genera los gráficos para la señal completa
generar_todos_los_graficos(resultados_totales, nombre_escenario="escenario2")

# --------- Detección por ventana deslizante (Escenario 2 mejorado) ---------
# Duración de la ventana de análisis y su paso, ambos en segundos
ventana_segundos = 5
paso_segundos = 1

# Conversión a cantidad de muestras
ventana_muestras = int(fs * ventana_segundos)
paso_muestras = int(fs * paso_segundos)
num_muestras = senal.shape[1]

# Cálculo del umbral de varianza (más estricto) basado en la distribución "before"
mean_var = resultados["before"]["canal_a_canal"]["Varianza"].mean()
std_var = resultados["before"]["canal_a_canal"]["Varianza"].std()
umbral_varianza = mean_var + 3 * std_var

print(f"\n>> Umbral de varianza usado (3σ): {umbral_varianza:.2f}")

# Inicialización de variables para tracking
detected = False
varianza_p80_ventanas = [] # Guarda percentil 80 por ventana
tiempos = []  # Marca de tiempo para cada ventana
detecciones = [] # Tiempos donde se detectó varianza sospechosa

# Iteración con ventana deslizante
for inicio in range(0, num_muestras - ventana_muestras, paso_muestras):
    fin = inicio + ventana_muestras
    segmento = senal[:, inicio:fin]
    segmento_centrado = segmento - np.mean(segmento, axis=1, keepdims=True)

    # Calcula varianza por canal, y obtiene el percentil 80 entre canales
    varianzas = np.var(segmento_centrado, axis=1)
    var_p80 = np.percentile(varianzas, 80)
    varianza_p80_ventanas.append(var_p80)

    # Guarda el tiempo correspondiente a esta ventana
    tiempo_actual = inicio / fs
    tiempos.append(tiempo_actual)

    # Detecta si el percentil 80 supera el umbral
    if not detected and var_p80 > umbral_varianza:
        detecciones.append(tiempo_actual)
        detected = True

# Muestra el resultado de la detección Mientras menor sea el retardo de deteccion mas preciso es el analisis
if detecciones:
    print(f"\n>> Se detectó actividad sospechosa a los {detecciones[0]:.2f} segundos")
    print(f">> Tiempo real de inicio de crisis: {t_ini} s")
    retardo = detecciones[0] - t_ini
    print(f">> Retardo de detección estimado: {retardo:.2f} s")
else:
    print("\n>> No se detectó ningún evento sobre el umbral.")

# ------------------ Gráfico de evolución de la varianza ------------------
plt.figure(figsize=(12, 6))
plt.axvspan(0, t_ini, color='green', alpha=0.1, label='Before')
plt.axvspan(t_ini, t_fin, color='red', alpha=0.1, label='Seizure')
plt.axvspan(t_fin, tiempos[-1], color='orange', alpha=0.1, label='After')
plt.plot(tiempos, varianza_p80_ventanas, label="Percentil 80 de varianza por ventana")
plt.axhline(umbral_varianza, color='red', linestyle='--', label="Umbral (3σ)")
plt.axvline(t_ini, color='green', linestyle='--', label="Inicio de crisis")
if detecciones:
    plt.axvline(detecciones[0], color='orange', linestyle='--', label="Detección")
plt.xlabel("Tiempo [s]")
plt.ylabel("Varianza (percentil 80)")
plt.title("Detección de crisis por varianza (mejorada)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("figuras", exist_ok=True)
plt.savefig("figuras/deteccion_varianza_mejorada.png")
plt.close()
