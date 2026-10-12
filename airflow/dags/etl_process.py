"""ETL process DAG for downloading, cleaning, and preprocessing healthcare stroke prediction dataset."""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

@dag(
    dag_id="etl_process",
    schedule=timedelta(days=15),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["etl"],
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

        s3 = boto3.client("s3")
        filename = "healthcare-dataset-stroke-data.csv"
        s3.upload_file(os.path.join(path, filename), "data", os.path.join("raw", filename))

        return f"s3://data/raw/{filename}"

    @task.virtualenv(requirements=["pandas", "boto3"], system_site_packages=False)
    def clean_data(s3_path):
        """Clean the downloaded data and perform feature engineering."""
        import pandas as pd
        import boto3
        import io

        s3 = boto3.client("s3")

        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = pd.read_csv(io.BytesIO(obj["Body"].read()))

        df = raw.copy()

        # Data cleaning: handle missing values, outliers, and irrelevant columns
        bmi_median = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(bmi_median)
        df = df[df["bmi"] <= 60]
        df = df.drop(columns=["id"])  ## Eliminamos la columna ID ya que no aporta informacion

        # Feature engineering: create new features based on existing ones
        df["high_glucose_level"] = (df["avg_glucose_level"] > 126).astype(int)
        df["high_bmi"] = (df["bmi"] > 30).astype(int)

        # Save data
        filename = "healthcare-dataset-stroke-data.csv"
        key = os.path.join("clean", filename)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        s3.put_object(Bucket="data", Key=key, Body=buf.getvalue())

        return f"s3://data/{key}"

    @task.virtualenv(requirements=["boto3", "pandas", "scikit-learn"], system_site_packages=False)
    def split_data(cleaned_s3_path):
        """Split the cleaned data into training, test, and calibration sets."""
        import pandas as pd
        import boto3
        import io
        from sklearn.model_selection import train_test_split

        s3 = boto3.client("s3")
        bucket, key = cleaned_s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        clean = pd.read_csv(io.BytesIO(obj["Body"].read()))

        # Split the data into training, test and calibration sets, stratifying by the target variable
        X = clean.drop(columns=["stroke"])
        y = clean["stroke"]

        X_train, X_test_calib, y_train, y_test_calib = train_test_split(
            X, y,
            test_size=0.3,  # 30% para test y calibracion
            stratify=y
        )

        X_test, X_calib, y_test, y_calib = train_test_split(
            X_test_calib, y_test_calib,
            test_size=0.5,  # repartimos mitad y mitad entre test y calibracion
            stratify=y_test_calib
        )

        # Save splits to S3
        splits = {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "calibration": (X_calib, y_calib),
        }

        for split_name, (X_split, y_split) in splits.items():
            split_df = pd.concat([X_split, y_split], axis=1)
            filename = f"healthcare-dataset-stroke-data-{split_name}.csv"
            key = os.path.join("splits", filename)
            buf = io.StringIO()
            split_df.to_csv(buf, index=False)
            s3.put_object(Bucket="data", Key=key, Body=buf.getvalue())

        return {split_name: f"s3://data/{key}" for split_name in splits.keys()}


    downloaded = download_data()
    cleaned = clean_data(downloaded)
    splits = split_data(cleaned)

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training",
        wait_for_completion=False,
    )

    splits >> trigger_training

etl_process()
