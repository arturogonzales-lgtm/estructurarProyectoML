# estructurarProyectoML

Proyecto de Machine Learning estructurado con un pipeline modular en Python.

## Fase 1: Configuración del Entorno

1. Crear la estructura de carpetas
   - Establecer un repositorio con archivos y módulos separados por responsabilidad.
   - Ejemplo de estructura:
     - `main.py`
     - `preprocessing.py`
     - `training.py`
     - `monitoring.py`
     - `requirements.txt`
     - `README.md`

2. Crear el entorno virtual e instalar dependencias
   - Crear el entorno virtual:
     - `python -m venv .venv`
   - Activar el entorno virtual:
     - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
   - Instalar las dependencias:
     - `pip install -r requirements.txt`

3. Configurar linter y formateador
   - Usar herramientas como `flake8` y `black` para mantener el código limpio y consistente.
   - Agregar configuración opcional en archivos como `.flake8`, `pyproject.toml` o `setup.cfg`.

## Fase 2: Integración del Dataset y Pipeline

1. Descripción del Dataset
   - Dataset: Telco Customer Churn
   - Fuente: WA_Fn-UseC_-Telco-Customer-Churn.csv
   - Descripción: Datos de clientes de telecomunicaciones para predecir churn (abandono).
   - Características: Información demográfica, servicios contratados, cargos, etc.

2. Pipeline de Machine Learning
   - Carga de datos: Lectura del CSV y limpieza inicial.
   - Preprocesamiento: Codificación de variables categóricas, escalado de numéricas, manejo de valores faltantes.
   - Monitoreo: Reportes de estadísticas de datos crudos y procesados.
   - Entrenamiento: Modelo Random Forest con hiperparámetros específicos.
   - Evaluación: Accuracy y reporte de clasificación.

3. Cómo ejecutar
   - Ejecutar el script principal: `python main.py`
   - El pipeline se ejecuta de manera secuencial, mostrando salidas formateadas en la terminal.

4. Resultados
   - Accuracy del modelo: ~81.37%
   - Reporte de clasificación incluye precision, recall y f1-score para clases 'No' y 'Yes'.

## Estado actual del proyecto

- `requirements.txt` contiene las dependencias necesarias.
- El repositorio ya incluye los módulos principales para el pipeline de ML.
- Código completo y modular.
- Cumple con PEP8 (linting con flake8 y formateo con black).
- Pipeline funcionando end-to-end con dataset integrado.




