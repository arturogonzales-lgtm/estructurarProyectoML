"""Módulo de entrenamiento y evaluación del modelo."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

TARGET_COL = "Churn"


def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 50):
    """Divide en features (X) y etiqueta (y), luego train/test."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_model() -> RandomForestClassifier:
    """Instancia y retorna el modelo Random Forest."""
    return RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_leaf_nodes=30,
    )


def train(df: pd.DataFrame):
    """Entrena el modelo y retorna el modelo y las particiones de prueba."""
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    metrics = {"X_test": X_test, "y_test": y_test}
    return model, metrics


def evaluate(model, metrics: dict) -> None:
    """Imprime las métricas de evaluación del modelo."""
    y_pred = model.predict(metrics["X_test"])
    print("\n" + "#" * 70)
    print("RANDOM FOREST - RESULTADO DE EVALUACIÓN")
    print("#" * 70)
    print(f"Accuracy: {accuracy_score(metrics['y_test'], y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(metrics["y_test"], y_pred))
    print("#" * 70)
