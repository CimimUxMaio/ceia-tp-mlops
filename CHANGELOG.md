# Changelog

Cambios realizados desde la entrega parcial `007f665` hasta la entrega final.

## Entrega final - 2026-10-18

### ETL y datos

- Se completó el DAG `etl_process`.
- Se implementaron las tareas de limpieza y generación de particiones.
- Se generan los conjuntos `train`, `test` y `calibration`.
- Se almacenan los datos y artefactos en MinIO.
- El DAG ETL dispara automáticamente el DAG `training`.
- Se corrigieron imports y configuraciones para ejecutar las tareas en entornos virtuales de Airflow.

### Entrenamiento

- Se reemplazaron las tareas de prueba por entrenamiento real.
- Se incorporaron modelos de Logistic Regression, Random Forest y Gradient Boosting.
- Se agregó búsqueda de hiperparámetros mediante `GridSearchCV`.
- Se incorporó balanceo de clases mediante SMOTE.
- Se agregó ajuste del umbral de decisión.
- Se calculan F1, recall, precision, balanced accuracy y ROC AUC.
- Se fijan semillas aleatorias por ejecución para favorecer la reproducibilidad.
- El preprocesamiento se integra al pipeline final junto con el estimador.
- El pipeline de preprocesamiento se persiste en MinIO para reutilizarlo durante las predicciones.

### MLflow

- Se implementó un run padre por ejecución del DAG `training`.
- Cada familia de modelos se registra como sub-run.
- Se registran parámetros, métricas y artefactos de cada modelo.
- Se registra el mejor modelo bajo `stroke-classifier` cuando supera la versión registrada anteriormente.
- Se agregaron tareas para finalizar los runs como `FINISHED` o `FAILED`.

### API

- Se implementó el endpoint `GET /health`.
- Se implementó `GET /versions/` para consultar las versiones registradas.
- Se implementó `POST /latest/` para predecir utilizando la última versión disponible.
- Se implementó `POST /versions/{version}/` para predecir utilizando una versión específica.

### Documentación y operación

- Se actualizó `DOC.md` con la arquitectura, los DAGs, MLflow y el contrato de la API.
- Se reemplazó el README del template por una guía del proyecto de predicción de ACV.
- Se documentaron los pasos de ejecución, verificación y limpieza.
