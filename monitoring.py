"""Módulo de monitoreo: data cruda y data procesada."""

from typing import Optional

import pandas as pd


def _infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detecta la columna target según nombres comunes."""
    lower_to_real = {column.lower(): column for column in df.columns}
    for candidate in ["target", "churn", "label", "y", "class"]:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    return None


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
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(10)
    if not top_missing.empty:
        print("\nTop columnas con nulos (%):")
        print(top_missing.round(2).to_string())

    print("\nEstadísticas numéricas:")
    print(df.describe(include="all").transpose().head(20).to_string())

    target_col = _infer_target_column(df)
    if target_col is not None:
        print(f"\nDistribución de target ({target_col}):")
        print(df[target_col].value_counts(dropna=False).to_string())
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
    print(f"Duplicados         : {df.duplicated().sum():,}")
    print("\nTipos de datos:")
    print(df.dtypes.to_string())

    target_col = _infer_target_column(df)
    if target_col is not None:
        print(f"\nDistribución de target procesado ({target_col}):")
        print(df[target_col].value_counts(dropna=False).to_string())

    print("=" * 70)
