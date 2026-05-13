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

## Estado actual del proyecto

- `requirements.txt` contiene las dependencias necesarias.
- El repositorio ya incluye los módulos principales para el pipeline de ML.

