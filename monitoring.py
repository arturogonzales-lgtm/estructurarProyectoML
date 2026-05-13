"""Módulo de monitoreo: data cruda y data procesada."""

import pandas as pd


# -- Monitoreo data CRUDA -----------------------------
def monitor_raw(df: pd.DataFrame) -> None:
    """
    Analiza la calidad de la data cruda.
    Reporta: forma, nulos, duplicados, KS y estadísticas básicas.
    """
    print("\n" + "=" * 70)
    print("MONITOREO — DATA CRUDA")
    print("=" * 70)
    print(f"Dataset original: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    print(f"Valores nulos totales : {df.isnull().sum().sum():,}")
    print(f"Duplicados totales   : {df.duplicated().sum():,}")
    print("\nEstadísticas numéricas:")
    print(df.describe().to_string())
    print("=" * 70)


# -- Monitoreo data PROCESADA -------------------------
def monitor_processed(df: pd.DataFrame) -> None:
    """
    Verifica la calidad de la data post-preprocesamiento.
    Reporta: forma, nulos restantes, KS y distribución de columnas.
    """
    print("\n" + "=" * 70)
    print("MONITOREO — DATA PROCESADA")
    print("=" * 70)
    print(f"Dataset procesado: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    print(f"Nulos restantes    : {df.isnull().sum().sum():,}")
    print("\nTipos de datos:")
    print(df.dtypes.to_string())
    print("=" * 70)
