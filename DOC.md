# Plataforma MLOps

## Descripcion general

La plataforma implementa un flujo completo para preparar datos, entrenar y
registrar modelos de machine learning, y exponer el modelo seleccionado para
realizar predicciones. El sistema esta compuesto por una API REST desarrollada
con FastAPI, MLflow como plataforma de experimentacion y gestion del ciclo de
vida de modelos, y Apache Airflow como orquestador de los procesos de datos y
entrenamiento.

Los servicios se ejecutan como una arquitectura distribuida. MinIO proporciona
el almacenamiento de objetos para datos y artefactos, PostgreSQL almacena la
informacion persistente de MLflow y Airflow, y Valkey funciona como soporte de
mensajeria para la ejecucion distribuida de Airflow.

## Arquitectura

El siguiente diagrama muestra los componentes principales y sus relaciones,
sin detallar la implementacion interna de cada servicio.

```mermaid
flowchart LR
    usuario[Usuario o cliente]
    fuente[Fuente externa de datos]
    api[FastAPI\nAPI REST de predicciones]
    mlflow[MLflow\nTracking y Model Registry]
    airflow[Apache Airflow\nOrquestacion de procesos]
    minio[MinIO\nAlmacenamiento de objetos]
    postgres[(PostgreSQL\nMetadatos y persistencia)]
    valkey[(Valkey\nMensajeria de Airflow)]

    usuario -->|Predicciones| api
    usuario -->|Gestion y ejecucion manual| airflow
    usuario -->|Experimentos, modelos y metricas| mlflow
    fuente -->|Datos de origen| airflow
    airflow -->|Ejecucion de ETL y training| minio
    airflow -->|Registro de runs y modelos| mlflow
    api -->|Carga del modelo y consulta del champion| mlflow
    mlflow -->|Artefactos| minio
    mlflow -->|Metadatos| postgres
    airflow -->|Metadatos| postgres
    airflow <--> |Tareas distribuidas| valkey
```

### Interfaces de usuario

- **FastAPI** expone una API REST y su documentacion interactiva mediante
  Swagger UI.
- **MLflow** proporciona una interfaz web para consultar experimentos, runs,
  metricas, artefactos y versiones registradas.
- **Airflow** proporciona una interfaz web para observar DAGs, revisar sus
  ejecuciones, consultar logs y lanzar procesos manualmente.
- **MinIO** proporciona una interfaz web para inspeccionar los buckets de
  datos y artefactos.

## FastAPI

FastAPI es la puerta de entrada para consumir los modelos registrados en
MLflow. La API recibe un conjunto de features en formato JSON, resuelve el
modelo solicitado y devuelve la prediccion generada.

La API publica los siguientes endpoints:

| Metodo | Endpoint | Descripcion |
| --- | --- | --- |
| `GET` | `/models` | Retorna el listado de modelos disponibles. |
| `POST` | `/models/champion/predict` | Predice utilizando el modelo que actualmente tiene el alias `champion`. |
| `POST` | `/models/{id}/predict` | Predice utilizando el modelo identificado por `{id}` en MLflow. |

El identificador `champion` es un caso especial. En cada solicitud, la API
consulta el modelo que ocupa ese alias en MLflow, por lo que la prediccion usa
el modelo con mejor resultado seleccionado por el proceso de entrenamiento.
El endpoint parametrizado permite solicitar de forma explicita otro modelo
registrado.

Para consultar el listado de modelos disponibles se utiliza el endpoint `GET`:

```bash
curl -X GET "http://localhost:8800/models"
```

La estructura exacta del JSON depende de los features utilizados por el modelo
activo. Por ejemplo, una solicitud conceptual para el modelo champion es:

```bash
curl -X POST "http://localhost:8800/models/champion/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_1": 10.5,
      "feature_2": 3,
      "feature_3": "valor"
    }
  }'
```

Para consultar un modelo especifico se utiliza su identificador en la ruta:

```bash
curl -X POST "http://localhost:8800/models/modelo-ejemplo/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_1": 10.5,
      "feature_2": 3,
      "feature_3": "valor"
    }
  }'
```

La documentacion interactiva de FastAPI esta disponible en
`http://localhost:8800/docs` y permite inspeccionar los contratos de la API y
probar las solicitudes desde el navegador.

## MLflow

MLflow centraliza el seguimiento de experimentos y la gestion de los modelos.
Sus responsabilidades dentro del sistema son:

- Registrar los entrenamientos, parametros, metricas y artefactos producidos.
- Organizar cada ejecucion dentro de un experimento.
- Mantener el registro de los modelos y sus versiones.
- Identificar el modelo seleccionado como `champion` mediante alias y tags.
- Proporcionar a FastAPI el modelo que debe utilizar para generar
  predicciones.

Los artefactos de los experimentos y modelos se almacenan en MinIO, mientras
que la informacion de seguimiento y registro se persiste en PostgreSQL. MLflow
tambien ofrece una interfaz web para explorar la evolucion de las metricas,
comparar ejecuciones y consultar el modelo champion.

### Estructura de los runs de entrenamiento

Cada ejecucion del DAG `training` se registra como un run principal de MLflow.
Dentro de ese run se crea un sub-run para cada familia de modelos evaluada.
Esta estructura permite comparar modelos y sus busquedas de hiperparametros
manteniendo todo el proceso de entrenamiento agrupado.

```mermaid
flowchart TD
    run[Run principal de training]
    linear[Sub-run: Linear Regression]
    forest[Sub-run: Random Forest]
    otros[Sub-runs: otros modelos]
    metricas[Metricas y artefactos]
    champion[Modelo ganador\nalias: champion]

    run --> linear
    run --> forest
    run --> otros
    linear --> metricas
    forest --> metricas
    otros --> metricas
    metricas --> champion
```

## Airflow y orquestacion de procesos

Apache Airflow coordina las tareas de preparacion de datos y entrenamiento
mediante dos DAGs. Cada DAG puede observarse y ejecutarse desde la interfaz
web de Airflow.

El DAG `etl_process` se ejecuta automaticamente cada 15 dias, de acuerdo con
la frecuencia de actualizacion de la fuente de datos. El DAG `training` se
dispara automaticamente cuando finaliza correctamente `etl_process`. Ambos
DAGs tambien pueden ejecutarse manualmente a pedido de un usuario.

### DAG `etl_process`

Este DAG transforma los datos de origen en un conjunto preparado para el
entrenamiento y la evaluacion. El resultado se almacena en MinIO para que
`training` pueda consumirlo de forma reproducible.

```mermaid
flowchart TD
    inicio([Inicio de etl_process])
    descarga[Descargar datos desde una fuente externa]
    limpieza[Limpiar datos y preparar el conjunto]
    split[Separar datos en train y test]
    preprocesamiento[Preprocesar datos\nencoding, scaling y balanceo]
    almacenamiento[Almacenar datasets preparados]
    fin([ETL finalizado])

    inicio --> descarga
    descarga --> limpieza
    limpieza --> split
    split --> preprocesamiento
    preprocesamiento --> almacenamiento
    almacenamiento --> fin
```

Las etapas del DAG son:

1. **Descarga:** obtiene la version actual de los datos desde la fuente
   externa.
2. **Limpieza:** corrige inconsistencias y prepara los registros para las
   siguientes etapas.
3. **Split:** divide el conjunto en datos de entrenamiento y de prueba.
4. **Preprocesamiento:** aplica encoding, escalado y balanceo de acuerdo con
   las necesidades del problema.
5. **Almacenamiento:** guarda los conjuntos resultantes para su consumo por el
   DAG de entrenamiento.

### DAG `training`

Este DAG toma los datos preparados por `etl_process`, entrena y evalua varios
tipos de modelos, y registra en MLflow el modelo que obtiene las mejores
metricas.

```mermaid
flowchart TD
    inicio([Inicio de training])
    lectura[Leer datos de entrenamiento almacenados]
    modelos[Definir familias de modelos]
    entrenamiento[Entrenar cada modelo\ncon busqueda y optimizacion de hiperparametros]
    evaluacion[Evaluar cada modelo con datos de prueba]
    comparacion[Comparar metricas de todos los modelos]
    seleccion[Seleccionar el modelo ganador]
    registro[Registrar el champion en MLflow\ncon tags y alias]
    fin([Training finalizado])

    inicio --> lectura
    lectura --> modelos
    modelos --> entrenamiento
    entrenamiento --> evaluacion
    evaluacion --> comparacion
    comparacion --> seleccion
    seleccion --> registro
    registro --> fin
```

El entrenamiento de cada familia de modelos se registra dentro de su propio
sub-run de MLflow. El run principal contiene la ejecucion completa y permite
relacionar la lectura de datos, los experimentos, la evaluacion y la decision
final. Una vez seleccionado el mejor resultado, el DAG registra esa version en
MLflow, agrega sus tags descriptivos y actualiza el alias `champion`.

## Flujo de prediccion

El flujo de inferencia separa al consumidor de los detalles internos de
entrenamiento y registro:

```mermaid
sequenceDiagram
    participant C as Cliente
    participant F as FastAPI
    participant M as MLflow
    participant A as Modelo champion

    C->>F: POST /models/champion/predict con features JSON
    F->>M: Solicitar modelo asociado al alias champion
    M-->>F: Entregar modelo seleccionado
    F->>A: Ejecutar prediccion con los features
    A-->>F: Devolver resultado
    F-->>C: Respuesta JSON con la prediccion
```

De esta forma, el cliente siempre utiliza un contrato REST estable, mientras
que la version concreta del modelo puede cambiar cuando una nueva ejecucion de
`training` actualiza el alias `champion`.

## Persistencia y almacenamiento

- **MinIO:** conserva los datasets preparados, los artefactos de MLflow y los
  archivos asociados a los modelos.
- **PostgreSQL:** conserva los metadatos de experimentos y modelos de MLflow,
  junto con la informacion persistente de Airflow.
- **Valkey:** soporta la comunicacion necesaria para distribuir las tareas de
  Airflow entre sus componentes de ejecucion.

Estos servicios complementan a FastAPI, MLflow y Airflow, pero no exponen el
contrato funcional principal del sistema. La interaccion de usuarios se realiza
principalmente mediante la API REST y las interfaces web de los servicios.
