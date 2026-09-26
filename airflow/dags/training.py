"""Dummy training pipeline with parallel model experiments."""

from datetime import datetime, timedelta
from time import sleep

from airflow.decorators import dag, task


@dag(
    dag_id="training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["training", "mlflow", "dummy"],
)
def training():
    @task
    def start_mlflow_run():
        """Start the parent MLflow run for this training execution."""
        sleep(1)
        return {
            "run_id": "dummy-training-run",
            "train_data": "s3://data/prepared/train",
            "test_data": "s3://data/prepared/test",
        }

    @task
    def train_linear_regression(training_context):
        """Read train/test data and train a linear regression sub-run."""
        # Future implementation: read both datasets and create an MLflow sub-run.
        sleep(1)
        return {"model": "linear-regression", "metric": 0.74, "context": training_context}

    @task
    def train_random_forest(training_context):
        """Read train/test data and train a random forest sub-run."""
        # Future implementation: read both datasets and create an MLflow sub-run.
        sleep(1)
        return {"model": "random-forest", "metric": 0.81, "context": training_context}

    @task
    def train_gradient_boosting(training_context):
        """Read train/test data and train a gradient boosting sub-run."""
        # Future implementation: read both datasets and create an MLflow sub-run.
        sleep(1)
        return {"model": "gradient-boosting", "metric": 0.86, "context": training_context}

    @task
    def train_support_vector_machine(training_context):
        """Read train/test data and train a support vector machine sub-run."""
        # Future implementation: read both datasets and create an MLflow sub-run.
        sleep(1)
        return {"model": "support-vector-machine", "metric": 0.78, "context": training_context}

    @task
    def compile_results(model_results):
        """Select the best candidate and describe future champion registration."""
        sleep(1)
        best_model = max(model_results, key=lambda result: result["metric"])
        current_champion_metric = 0.82

        # Future implementation: if the candidate is better, register it in
        # MLflow and assign the champion alias.
        return {
            "selected_model": best_model["model"],
            "metric": best_model["metric"],
            "promote_to_champion": best_model["metric"] > current_champion_metric,
        }

    parent_run = start_mlflow_run()
    model_results = [
        train_linear_regression(parent_run),
        train_random_forest(parent_run),
        train_gradient_boosting(parent_run),
        train_support_vector_machine(parent_run),
    ]
    compile_results(model_results)


training()
