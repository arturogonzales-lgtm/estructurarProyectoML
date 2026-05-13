"""
TAREA: Pipeline de Machine Learning
=====================================
Rúbrica de evaluación:
 - Código completo (todas las etapas funcionando) : 10 puntos
 - Código ordenado (estructura modular clara) : 7 puntos
 - Buenas prácticas (funciones, docstrings, PEP8) : 3 puntos
"""

import preprocessing as pp
import monitoring as mn
import training as tr

# Reemplazar con la ruta real del dataset
DATA_PATH = "resources/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main():
    print("\n" + "#" * 70)
    print("INICIO DEL PROYECTO DE MACHINE LEARNING")
    print("#" * 70)
    print(f"Ruta del dataset: {DATA_PATH}\n")

    print("--- ETAPA 1: CARGA Y PREPROCESAMIENTO ---")
    df_raw = pp.load_data(DATA_PATH)
    df_processed = pp.preprocess(df_raw)

    print("\n--- ETAPA 2: MONITOREO DE DATOS ---")
    mn.monitor_raw(df_raw)
    mn.monitor_processed(df_processed)

    print("\n--- ETAPA 3: ENTRENAMIENTO RANDOM FOREST ---")
    model, metrics = tr.train(df_processed)
    tr.evaluate(model, metrics)


if __name__ == "__main__":
    main()
