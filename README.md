# Prediccion de riesgo de ACV
## Integrantes

- Martin Mauricio Villodas Maizon
- Juan Bautista Abdon
- Martin Ezequiel Radio Villanueva

Este proyecto implementa una plataforma MLOps para estimar el riesgo de que un
paciente haya sufrido un accidente cerebrovascular (ACV o *stroke*) a partir de
informacion demografica y clinica. La prediccion puede servir como apoyo para
analisis, investigacion o priorizacion de pacientes, pero no reemplaza una
evaluacion ni un diagnostico medico.

La plataforma automatiza el recorrido completo de los datos: descarga y
limpieza del dataset, generacion de particiones, entrenamiento de modelos,
registro de experimentos y modelos, y exposicion de predicciones mediante una
API REST.

## El problema

El proyecto utiliza el [Stroke Prediction Dataset de Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset),
que contiene 5110 registros de pacientes y la variable objetivo `stroke`.
Las observaciones incluyen, entre otras, las siguientes caracteristicas:

- Edad.
- Hipertension y enfermedad cardiaca.
- Nivel promedio de glucosa.
- Indice de masa corporal (BMI).
- Genero y estado civil.
- Tipo de trabajo y residencia.
- Estado de tabaquismo.

La clase positiva representa aproximadamente el 5% de los registros, por lo
que el desbalance de clases es una condicion importante del problema. El flujo
de datos contempla el tratamiento de valores faltantes de `bmi`, la exclusion
de valores extremos y la generacion de las features derivadas
`high_glucose_level` y `high_bmi`.

## Plataforma

El sistema esta compuesto por los siguientes servicios:

- **Apache Airflow:** orquesta el ETL y el entrenamiento.
- **MLflow:** registra experimentos, metricas, artefactos y versiones de modelos.
- **FastAPI:** expone el estado de la API, las versiones disponibles y las predicciones.
- **MinIO:** provee almacenamiento de objetos compatible con S3.
- **PostgreSQL:** persiste los metadatos de Airflow y MLflow.
- **Valkey:** funciona como broker para las tareas distribuidas de Airflow.

El flujo general es:

```text
Dataset -> ETL -> Splits -> Entrenamiento -> MLflow -> FastAPI
```

La documentacion completa de la arquitectura, los DAGs, el contrato de la API
y la estructura de los runs se encuentra en [DOC.md](DOC.md).

## Quick Start

### Requisitos

- Docker Engine o Docker Desktop.
- Docker Compose.
- GNU Make.

Los servicios se ejecutan mediante Docker Compose y estan configurados para
desarrollo local.

### Iniciar la plataforma

Desde la raiz del repositorio:

```bash
make start
```

Este target inicia todos los servicios en segundo plano y muestra las URLs
disponibles. Para volver a consultar las URLs:

```bash
make urls
```

Tambien se puede consultar la ayuda del Makefile:

```bash
make help
```

### Interfaces

| Servicio | URL | Uso |
| --- | --- | --- |
| Airflow | `http://localhost:8080` | Ejecutar y observar los DAGs |
| MLflow | `http://localhost:5001` | Consultar runs y modelos |
| MinIO Console | `http://localhost:9001` | Inspeccionar buckets y objetos |
| Frontend | `http://localhost:8800/` | Predecir desde un formulario web |
| FastAPI | `http://localhost:8800` | Consultar la API |
| Swagger UI | `http://localhost:8800/docs` | Probar los endpoints |

Las credenciales iniciales de Airflow son `airflow` / `airflow`, salvo que se
hayan cambiado en `.env`.

## Ejecutar el flujo

1. Ejecutar `make start`.
2. Abrir Airflow en `http://localhost:8080`.
3. Despausar los DAGs `etl_process` y `training`. Los DAGs se crean pausados;
   si `training` queda pausado, la ejecucion que dispara el ETL queda en cola
   y no se ejecuta. Al despausar `etl_process` tambien se activa su schedule
   de 15 dias.
4. Ejecutar manualmente el DAG `etl_process`.
5. Verificar que sus tareas se ejecuten en este orden:
   `download_data` -> `clean_data` -> `split_data` -> `trigger_training`.
6. Verificar que `trigger_training` inicie el DAG `training`.
7. Observar en `training` la creacion del preprocesador, el run padre de
   MLflow, los entrenamientos y la compilacion de resultados.
8. Consultar MinIO, MLflow y FastAPI para verificar los artefactos y resultados.

El ETL guarda en el bucket `data` los datos raw y clean, y los splits de
train, test y calibration. El DAG `training` guarda en el mismo bucket
`pipelines/preprocessor.pkl`, registra sus runs y artefactos en MLflow y
publica las versiones bajo el modelo `stroke-classifier` cuando corresponde.

## Verificaciones rapidas

Comprobar que FastAPI esta disponible:

```bash
curl http://localhost:8800/health
```

Consultar las versiones registradas:

```bash
curl http://localhost:8800/versions/
```

Las predicciones se pueden probar desde Swagger o mediante los endpoints
`POST /latest/` y `POST /versions/{version}/`. El esquema de entrada y los
ejemplos completos se encuentran en [DOC.md](DOC.md).

En MinIO se espera encontrar:

- `data/raw/`
- `data/clean/`
- `data/splits/`
- `data/pipelines/preprocessor.pkl`
- Artefactos de MLflow en el bucket `mlflow`.

En MLflow se espera encontrar el experimento `stroke-training`, sus runs y el
modelo registrado `stroke-classifier`.

## Detener y limpiar

Para detener los servicios sin eliminar los datos persistidos:

```bash
make stop
```

Para detener los servicios y eliminar imagenes, volumenes, logs y datos
persistidos:

```bash
make clean
```

Usar `make clean` solo cuando se quiera reiniciar completamente el ambiente.
En Windows, este target requiere un shell POSIX (por ejemplo Git Bash o WSL),
ya que utiliza `rm` y `find`.
