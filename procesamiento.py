from pyedflib import highlevel # Librería para lectura de archivos EDF (formato usado para EEG)
import numpy as np

def cargar_datos(archivo_edf, archivo_summary):
    # Lee la señal EEG desde el archivo .edf y sus metadatos
    signals,signal_headers,header = highlevel.read_edf(archivo_edf)

    # Extrae la frecuencia de muestreo desde los headers del archivo
    fs = signal_headers[0]['sample_frequency']  # frecuencia de muestreo (Hz)

    # Extrae los nombres de los canales (electrodos) del archivo
    canales = [h['label'] for h in signal_headers]
    
    # Lee el archivo .txt resumen (summary) y busca los tiempos de inicio y fin de crisis
    with open(archivo_summary, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            #Busca la sección correspondiente al archivo actual
            if "File Name" in line and archivo_edf.split('/')[-1] in line:

                for j in range(i, i+10): # Busca dentro de las 10 líneas siguientes
                    # Extrae el tiempo de inicio de la crisis (en segundos)
                    if "Seizure Start Time" in lines[j]:
                        t_ini = int(lines[j].split(":")[1].replace("seconds", "").strip())
                    # Extrae el tiempo de fin de la crisis (en segundos)
                    if "Seizure End Time" in lines[j]:
                        t_fin = int(lines[j].split(":")[1].replace("seconds", "").strip())
                break # Ya encontró la sección correspondiente, sale del bucle

    # Devuelve la señal como un array de NumPy, frecuencia de muestreo, nombres de canales y tiempos de crisis
    return np.array(signals), fs, canales, t_ini, t_fin

def segmentar_senal(senal, fs, t_ini, t_fin):
    # Convierte los tiempos de interés a índices de muestra (para extraer porciones de señal)
    
    muestras_inicio = int((t_ini - 120) * fs)  # 2 minutos antes de la crisis
    muestras_crisis_inicio = int(t_ini * fs) # inicio de la crisis
    muestras_crisis_fin = int(t_fin * fs) # fin de la crisis
    muestras_fin = int((t_fin + 120) * fs) # 2 minutos después de la crisis

    # Extrae las tres secciones de interés de la señal: before, seizure y after
    before = senal[:, muestras_inicio:muestras_crisis_inicio]
    seizure = senal[:, muestras_crisis_inicio:muestras_crisis_fin]
    after = senal[:, muestras_crisis_fin:muestras_fin]

    # Centrado de señal: se resta la media por canal (para eliminar componente DC)
    before = before - np.mean(before, axis=1, keepdims=True)
    seizure = seizure - np.mean(seizure, axis=1, keepdims=True)
    after = after - np.mean(after, axis=1, keepdims=True)

    # Devuelve los tres bloques segmentados
    return {"before": before, "seizure": seizure, "after": after}
