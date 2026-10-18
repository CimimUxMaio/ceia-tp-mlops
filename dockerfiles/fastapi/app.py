import os
from enum import Enum

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

MODEL_NAME = "stroke-classifier"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

app = FastAPI()

_client = mlflow.MlflowClient()
_model_cache = {}


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class EverMarried(str, Enum):
    YES = "Yes"
    NO = "No"


class WorkType(str, Enum):
    CHILDREN = "children"
    GOVT_JOB = "Govt_job"
    NEVER_WORKED = "Never_worked"
    PRIVATE = "Private"
    SELF_EMPLOYED = "Self-employed"


class ResidenceType(str, Enum):
    URBAN = "Urban"
    RURAL = "Rural"


class SmokingStatus(str, Enum):
    FORMERLY_SMOKED = "formerly smoked"
    NEVER_SMOKED = "never smoked"
    SMOKES = "smokes"
    UNKNOWN = "Unknown"


class PatientFeatures(BaseModel):
    gender: Gender
    age: float
    hypertension: int
    heart_disease: int
    ever_married: EverMarried
    work_type: WorkType
    Residence_type: ResidenceType
    avg_glucose_level: float
    bmi: float
    smoking_status: SmokingStatus


def _to_dataframe(features: PatientFeatures) -> pd.DataFrame:
    """Build the model input, replicating the ETL feature engineering."""
    data = features.model_dump()
    data["high_glucose_level"] = int(data["avg_glucose_level"] > 126)
    data["high_bmi"] = int(data["bmi"] > 30)
    return pd.DataFrame([data])


def _get_model(version: str):
    if version not in _model_cache:
        try:
            _model_cache[version] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        except MlflowException as error:
            raise HTTPException(status_code=404, detail=f"Model version '{version}' not found") from error
    return _model_cache[version]


def _get_latest_version() -> str:
    versions = _client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise HTTPException(status_code=404, detail=f"No versions registered for model '{MODEL_NAME}'")
    return max(versions, key=lambda model_version: int(model_version.version)).version


def _predict(version: str, features: PatientFeatures):
    model = _get_model(version)
    df = _to_dataframe(features)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]
    return {
        "version": version,
        "prediction": int(prediction),
        "probability": float(probability),
    }


@app.get("/", include_in_schema=False)
def index():
    """Serve the web frontend."""
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health_check():
    """Report that the API process is available."""
    return {"status": "ok"}




@app.get("/versions/")
def list_versions():
    """Return all registered versions of the stroke classifier."""
    versions = _client.search_model_versions(f"name='{MODEL_NAME}'")
    return [
        {
            "version": version.version,
            "run_id": version.run_id,
            "status": version.status,
            "creation_timestamp": version.creation_timestamp,
            "tags": version.tags,
        }
        for version in sorted(versions, key=lambda v: int(v.version), reverse=True)
    ]


@app.post("/latest/")
def predict_latest(features: PatientFeatures):
    """Predict using the most recently registered model version."""
    return _predict(_get_latest_version(), features)


@app.post("/versions/{version}/")
def predict_version(version: str, features: PatientFeatures):
    """Predict using a specific registered model version."""
    return _predict(version, features)
