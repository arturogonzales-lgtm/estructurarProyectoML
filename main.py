"""
TAREA: Pipeline de Machine Learning
=====================================
Rúbrica de evaluación:
 - Código completo (todas las etapas funcionando) : 10 puntos
 - Código ordenado (estructura modular clara) : 7 puntos
 - Buenas prácticas (funciones, docstrings, PEP8) : 3 puntos
"""

import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import preprocessing as pp
import monitoring as mn
import training as tr

# Reemplazar con la ruta real del dataset
DATA_PATH = "resources/data/Data_CU_venta.csv"


class Tee:
    """Duplica la salida hacia múltiples streams (terminal + archivo)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


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
    log_dir = Path("resources/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_output.txt"

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = Tee(sys.__stdout__, log_file)
        stderr_tee = Tee(sys.__stderr__, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            main()

    print(f"\nSalida del pipeline guardada en: {log_path}")
