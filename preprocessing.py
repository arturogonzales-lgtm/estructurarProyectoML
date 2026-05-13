"""Módulo de preprocesamiento de datos."""
import pandas as pd
import numpy as np
def load_data(path: str) -> pd.DataFrame:
 """Carga el dataset desde la ruta indicada."""
 df = pd.read_csv(path) # ajustar formato si es necesario
 return df
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
 """Imputa o elimina valores nulos."""
 df = df.dropna() # o df.fillna(...)
 return df
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
 """Codifica variables categóricas."""
 # Ejemplo: pd.get_dummies(df, columns=[...])
 return df
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
 """Escala variables numéricas."""
 # Ejemplo: StandardScaler / MinMaxScaler
 return df
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
 """Pipeline completo de preprocesamiento."""
 df = handle_missing(df)
 df = encode_features(df)
 df = scale_features(df)
 return df