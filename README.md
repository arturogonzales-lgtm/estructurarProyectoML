# estructurarProyectoML

Pipeline de Machine Learning modular en Python para procesamiento de datos tabulares,
monitoreo de calidad y entrenamiento de un modelo de clasificación con Random Forest.

Este proyecto fue adaptado para trabajar con el dataset:

- `resources/data/Data_CU_venta.csv`

## Objetivo

Ejecutar un flujo end-to-end reproducible que incluya:

1. Carga de datos.
2. Preprocesamiento robusto.
3. Monitoreo de calidad de datos (antes y después del procesamiento).
4. Entrenamiento de modelo supervisado.
5. Evaluación con métricas de clasificación.

## Estructura del proyecto

```text
.
|-- main.py
|-- preprocessing.py
|-- monitoring.py
|-- training.py
|-- requirements.txt
|-- README.md
`-- resources/
    |-- data/
        `-- Data_CU_venta.csv
```

## Flujo del pipeline

### 1) Orquestación (`main.py`)

`main.py` ejecuta las etapas en este orden:

1. Carga y preprocesamiento.
2. Monitoreo de datos crudos y procesados.
3. Entrenamiento y evaluación.

Ruta activa del dataset:

```python
DATA_PATH = "resources/data/Data_CU_venta.csv"
```

### 2) Preprocesamiento (`preprocessing.py`)

El módulo aplica una lógica robusta para datasets tabulares, inspirada en el notebook
`resources/notebook/1_Preprocessing.ipynb`, pero automatizada para uso productivo.

#### Reglas aplicadas

1. Limpieza base:
   - Elimina `customerID` si existe.
   - Limpia espacios en columnas categóricas.
2. Conversión automática de texto a numérico:
   - Si una columna tipo `object` tiene >= 90% de valores numéricos válidos,
     se convierte a numérica.
3. Eliminación por nulos:
   - Elimina columnas con más de 80% de valores faltantes.
4. Reglas de dominio del nuevo dataset (si existen columnas):
   - `seg_un`: reemplaza valor `3` por `0`.
   - `grp_riesgociiu`: agrupa `grupo_2`, `grupo_3`, `grupo_9`, `grupo_8`,
     `grupo_1` en `grupo_11`.
5. Imputación de nulos:
   - Numéricas: mediana (o `0` si la mediana no es calculable).
   - Categóricas: `"Otros"`.
6. Encoding:
   - Codifica variables categóricas de entrada con `LabelEncoder`.
   - No codifica la columna target en esta etapa.
7. Escalado:
   - Estandariza columnas numéricas con `StandardScaler`.
   - Excluye la columna objetivo y `p_codmes` del escalado.

#### Detección automática de target

El pipeline detecta como objetivo cualquiera de estos nombres (case-insensitive):

- `target`, `churn`, `label`, `y`, `class`

### 3) Monitoreo (`monitoring.py`)

Se generan reportes para datos crudos y procesados.

#### En data cruda

- Tamaño del dataset.
- Nulos totales.
- Duplicados totales.
- Top de columnas con mayor porcentaje de nulos.
- Estadísticos descriptivos (`describe(include="all")`, vista resumida).
- Distribución de la variable objetivo (si se detecta).

#### En data procesada

- Tamaño del dataset procesado.
- Nulos restantes.
- Duplicados.
- Tipos de datos finales.
- Distribución de target procesado.

### 4) Entrenamiento y evaluación (`training.py`)

#### Preparación

1. Detección automática de la columna objetivo.
2. Separación de `X` e `y`.
3. Codificación de `y` con `LabelEncoder` solo si es categórica.

#### Split

- `train_test_split(test_size=0.3, random_state=50)`
- Con `stratify=y` cuando hay más de una clase.

#### Modelo

`RandomForestClassifier` con:

- `n_estimators=300`
- `oob_score=True`
- `n_jobs=-1`
- `random_state=50`
- `max_leaf_nodes=30`
- `min_samples_leaf=1`
- `max_features="sqrt"`

#### Métricas

- `accuracy`
- `classification_report`

## Configuración del entorno

### Requisitos

- Python 3.10+ (recomendado)
- PowerShell (en Windows)

### Instalación

1. Crear entorno virtual:

```powershell
python -m venv .venv
```

2. Activar entorno virtual (Windows PowerShell):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecución

Con el entorno activo, ejecutar:

```powershell
python main.py
```

La salida en consola mostrará:

1. Inicio del pipeline y ruta del dataset.
2. Reporte de monitoreo en data cruda.
3. Reporte de monitoreo en data procesada.
4. Métricas finales del modelo.

## Notebook de referencia

El notebook `resources/notebook/1_Preprocessing.ipynb` se usa como referencia lógica
de negocio para el preprocesamiento del nuevo dataset. El código productivo de este
repositorio implementa esa lógica de forma automatizada y mantenible en módulos Python.

## Errores comunes y solución

1. No se encuentra la columna objetivo:
   - Verificar que el dataset tenga una columna llamada `target`, `churn`, `label`, `y` o `class`.
2. Error al leer CSV:
   - Confirmar que existe `resources/data/Data_CU_venta.csv`.
3. Dependencias faltantes:
   - Ejecutar de nuevo `pip install -r requirements.txt` con el entorno activado.

## Estado actual

- Proyecto modular y ejecutable end-to-end.
- Pipeline actualizado para `Data_CU_venta.csv`.
- Lógica de preprocesamiento, monitoreo y entrenamiento alineada al nuevo caso.




