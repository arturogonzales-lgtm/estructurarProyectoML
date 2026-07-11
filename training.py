"""Módulo de entrenamiento y evaluación del modelo."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "target"   # <-- reemplazar con columna objetivo real
TARGET_CANDIDATES = [TARGET_COL, "churn", "label", "y", "class"]
# -- eliminar si no se desea inferir automáticamente la columna objetivo


def infer_target_column(df: pd.DataFrame) -> str:
    """Detecta la variable objetivo en base a nombres comunes."""
    lower_to_real = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    raise ValueError(
        "No se encontró columna objetivo. "
        "Se esperaba una de: target, churn, label, y, class"
    )


def prepare_xy(df: pd.DataFrame):
    """Prepara matriz de features y vector objetivo."""
    target_col = infer_target_column(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    target_encoder = None
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    return X, y, target_col, target_encoder


def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 50):
    """Divide en features (X) y etiqueta (y), luego train/test."""
    X, y, target_col, target_encoder = prepare_xy(df)
    stratify = y if pd.Series(y).nunique() > 1 else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    ), target_col, target_encoder


def build_model() -> RandomForestClassifier:
    """Instancia y retorna el modelo Random Forest."""
    return RandomForestClassifier(
        n_estimators=300,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_leaf_nodes=30,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
    )


def train(df: pd.DataFrame):
    """Entrena el modelo y retorna el modelo y las particiones de prueba."""
    (X_train, X_test, y_train, y_test), target_col, target_encoder = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    metrics = {
        "X_test": X_test,
        "y_test": y_test,
        "target_col": target_col,
        "target_encoder": target_encoder,
    }
    return model, metrics


def evaluate(model, metrics: dict) -> None:
    """Imprime las métricas de evaluación del modelo."""
    try:
        y_pred = model.predict(metrics["X_test"])
    except NotFittedError as exc:
        raise RuntimeError(
            "El modelo no está entrenado. Ejecuta train() antes."
        ) from exc

    print("\n" + "#" * 70)
    print("RANDOM FOREST - RESULTADO DE EVALUACIÓN")
    print("#" * 70)
    print(f"Target detectado: {metrics.get('target_col')}")
    print(f"Accuracy: {accuracy_score(metrics['y_test'], y_pred):.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(metrics["y_test"], y_pred))
    print("#" * 70)
