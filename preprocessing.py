"""Módulo de preprocesamiento de datos."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Churn"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta indicada."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza inicial y normalización básica del dataset."""
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "tenure" in df.columns:
        df = df[df["tenure"] != 0].copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas y la variable objetivo."""
    object_columns = df.select_dtypes(include=["object", "category"]).columns
    encoder = LabelEncoder()
    for column in object_columns:
        df[column] = encoder.fit_transform(df[column].astype(str))
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Escala las columnas numéricas usando StandardScaler."""
    scaler = StandardScaler()
    numeric_columns = [col for col in NUM_COLS if col in df.columns]
    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns].astype(float))
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preprocesamiento para el dataset de churn."""
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    return df
