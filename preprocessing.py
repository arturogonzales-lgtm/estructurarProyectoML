"""Módulo de preprocesamiento de datos."""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_CANDIDATES = ["target", "churn", "label", "y", "class"]
HIGH_NAN_THRESHOLD = 80.0


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta indicada."""
    return pd.read_csv(path)


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detecta la columna objetivo usando nombres comunes (sin importar mayúsculas)."""
    lower_to_real = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    return None


def drop_high_nan_columns(
    df: pd.DataFrame, threshold: float = HIGH_NAN_THRESHOLD
) -> pd.DataFrame:
    """Elimina columnas con más de ``threshold`` porcentaje de nulos."""
    nan_ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    columns_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def coerce_numeric_like_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a numéricas las columnas object que son mayormente números."""
    object_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in object_columns:
        candidate = pd.to_numeric(df[column], errors="coerce")
        conversion_ratio = candidate.notna().mean()
        if conversion_ratio >= 0.9:
            df[column] = candidate
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza inicial orientada a datos tabulares de negocio."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Limpieza ligera de strings para evitar categorías duplicadas por espacios.
    object_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in object_columns:
        df[column] = (
            df[column].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        )

    df = coerce_numeric_like_objects(df)
    df = drop_high_nan_columns(df)
    return df


def apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas específicas observadas en el notebook del nuevo dataset."""
    if "seg_un" in df.columns:
        df["seg_un"] = pd.to_numeric(df["seg_un"], errors="coerce")
        df["seg_un"] = df["seg_un"].replace(3, 0)

    if "grp_riesgociiu" in df.columns:
        replace_groups = ["grupo_2", "grupo_3", "grupo_9", "grupo_8", "grupo_1"]
        df["grp_riesgociiu"] = df["grp_riesgociiu"].replace(replace_groups, "grupo_11")

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa nulos: mediana para numéricas y 'Otros' para categóricas."""
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    for column in numeric_columns:
        median_value = df[column].median()
        if pd.notna(median_value):
            df[column] = df[column].fillna(median_value)
        else:
            df[column] = df[column].fillna(0)

    for column in categorical_columns:
        df[column] = df[column].fillna("Otros")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas de entrada (excepto target)."""
    target_col = infer_target_column(df)
    object_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in object_columns:
        object_columns.remove(target_col)

    encoder = LabelEncoder()
    for column in object_columns:
        df[column] = encoder.fit_transform(df[column].astype(str))
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Escala columnas numéricas continuas (sin tocar target ni columnas temporales)."""
    target_col = infer_target_column(df)
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded = {
        column for column in [target_col, "p_codmes"] if column in numeric_columns
    }
    numeric_columns = [column for column in numeric_columns if column not in excluded]

    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns].astype(float))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preprocesamiento para datasets tabulares."""
    df = clean_data(df)
    df = apply_domain_rules(df)
    df = impute_missing_values(df)
    df = encode_features(df)
    df = scale_features(df)
    return df
