"""Dummy ETL pipeline for preparing data used by the training DAG."""

from datetime import datetime, timedelta
from time import sleep

from airflow.decorators import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="etl_process",
    schedule=timedelta(days=15),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["etl", "dummy"],
)
def etl_process():
    @task.virtualenv(requirements=["kagglehub", "boto3"], system_site_packages=False)
    def download_data():
        """Download the latest source data."""
        import kagglehub
        import boto3
        import os

        # Download latest version
        path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
        filename = "healthcare-dataset-stroke-data.csv"

        s3 = boto3.client("s3")
        s3.upload_file(os.path.join(path, filename), "data", os.path.join("raw", filename))

        return f"s3://data/raw/{filename}"

    @task
    def clean_data(source_data):
        """Clean inconsistent records before preparing the dataset."""
        sleep(1)
        return {"cleaned": True}

    @task
    def split_data(cleaned_data):
        """Split the cleaned data into training and testing datasets."""
        sleep(1)
        return {**cleaned_data, "train": "dummy-train-data", "test": "dummy-test-data"}

    @task
    def preprocess_data(datasets):
        """Apply encoding, scaling, and balancing to both datasets."""
        sleep(1)
        return {**datasets, "preprocessed": True}

    @task
    def store_datasets(preprocessed_datasets):
        """Store prepared datasets for consumption by the training DAG."""
        sleep(1)
        return {"location": "s3://data/prepared", "datasets": preprocessed_datasets}

    downloaded = download_data()
    cleaned = clean_data(downloaded)
    split = split_data(cleaned)
    preprocessed = preprocess_data(split)
    stored = store_datasets(preprocessed)

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training",
        wait_for_completion=False,
    )

    stored >> trigger_training


etl_process()
