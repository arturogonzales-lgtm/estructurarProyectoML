"""
TAREA: Pipeline de Machine Learning
=====================================
Rúbrica de evaluación:
 - Código completo (todas las etapas funcionando) : 10 puntos
 - Código ordenado (estructura modular clara) : 7 puntos
 - Buenas prácticas (funciones, docstrings, PEP8) : 3 puntos
Dataset: cargado desde DIRECCIÓN DRIVE
"""

import preprocessing as pp
import monitoring as mn
import training as tr
DATA_PATH = "DIRECCIÓN DRIVE" # <-- reemplazar con ruta real
def main():
 print("Iniciando el proyecto de ML")
 # 1. Cargar y preprocesar
 df_raw = pp.load_data(DATA_PATH)
 df_processed = pp.preprocess(df_raw)
 # 2. Monitoreo
 mn.monitor_raw(df_raw)
 mn.monitor_processed(df_processed)
 # 3. Entrenamiento
 model, metrics = tr.train(df_processed)
 tr.evaluate(model, metrics)
if __name__ == "__main__":
 main()