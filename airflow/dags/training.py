"""Training pipeline with parallel, parametrized model experiments."""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.utils.trigger_rule import TriggerRule


RANDOM_STATE = 42

MODEL_CONFIGS = [
    {
        "task_id": "train_logistic_regression",
        "model": "Logistic Regression",
        "estimator": "logistic_regression",
        "params": {"max_iter": 2000},
        "param_grid": {
            "model__estimator__classifier__C": {
                "type": "logspace",
                "start": -4,
                "stop": 4,
                "num": 10,
                "extra": [1.0],
            },
        },
    },
    {
        "task_id": "train_random_forest",
        "model": "Random Forest",
        "estimator": "random_forest",
        "params": {"class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1},
        "param_grid": {
            "model__estimator__classifier__n_estimators": [100, 300, 500],
            "model__estimator__classifier__max_depth": [None, 10, 30],
            "model__estimator__classifier__max_features": ["sqrt", "log2", 0.8, 0.5],
        },
    },
    {
        "task_id": "train_gradient_boosting",
        "model": "Gradient Boosting",
        "estimator": "gradient_boosting",
        "params": {"random_state": RANDOM_STATE},
        "param_grid": {
            "model__estimator__classifier__n_estimators": [100, 300, 500],
            "model__estimator__classifier__learning_rate": [0.01, 0.1, 0.3],
            "model__estimator__classifier__max_depth": [3, 5, 7],
        },
    },
]


@dag(
    dag_id="training",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["training", "mlflow", "dummy"],
)
def training():
    @task.virtualenv(requirements=["boto3", "scikit-learn", "feature-engine"], system_site_packages=False)
    def create_preprocessor():
        """Apply encoding, scaling, and balancing to both datasets."""
        import os
        import io

        import boto3
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler
        from feature_engine.encoding import CountFrequencyEncoder

        s3 = boto3.client("s3")

        # Encoding & Scaling
        binary_features = ["ever_married"]
        target_features = ["gender", "smoking_status", "work_type", "Residence_type"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("binary", OrdinalEncoder(), binary_features),
                ("frequency", CountFrequencyEncoder(encoding_method="frequency"), target_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(steps=[
            ("encoder", preprocessor),
            ("scaler", StandardScaler()),
        ])

        pipeline.set_output(transform="pandas")

        # Save preprocessor pipeline to S3
        key = os.path.join("pipelines", "preprocessor.pkl")
        buf = io.BytesIO()
        joblib.dump(pipeline, buf)
        buf.seek(0)
        s3.put_object(Bucket="data", Key=key, Body=buf.getvalue())

        return f"s3://data/{key}"

    @task
    def start_mlflow_run():
        """Start the parent MLflow run for this training execution."""
        import os

        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        experiment = mlflow.set_experiment("stroke-training")
        client = mlflow.MlflowClient()

        run = client.create_run(
            experiment_id=experiment.experiment_id,
        )
        return {
            "run_id": run.info.run_id,
            "experiment_id": experiment.experiment_id,
        }

    @task.virtualenv(
        requirements=[
            "boto3",
            "scikit-learn",
            "pandas",
            "joblib",
            "feature-engine",
            "imbalanced-learn",
            "mlflow>=3.15",
        ],
        system_site_packages=False,
    )
    def train_model(training_context, preprocessor_path, config, random_state):
        """Read train/test data, grid search a model, and evaluate the best one."""
        import io
        import os

        import boto3
        import joblib
        import mlflow
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
        from sklearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.metrics import (
            recall_score,
            precision_score,
            f1_score,
            balanced_accuracy_score,
            roc_auc_score,
        )

        def resolve_param_grid(param_grid):
            """Expand declarative search-space specs into concrete value lists."""
            resolved = {}

            for parameter, values in param_grid.items():
                if isinstance(values, dict) and values.get("type") == "logspace":
                    generated = np.logspace(
                        values["start"],
                        values["stop"],
                        values["num"],
                    )
                    resolved[parameter] = generated.tolist() + values.get("extra", [])
                else:
                    resolved[parameter] = values

            return resolved

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        child_run = client.create_run(
            experiment_id=training_context["experiment_id"],
            tags={
                "mlflow.runName": config["model"],
                "mlflow.parentRunId": training_context["run_id"],
                "model_type": config["model"],
            },
        )

        with mlflow.start_run(run_id=child_run.info.run_id):
            s3 = boto3.client("s3")

            estimator_classes = {
                "logistic_regression": LogisticRegression,
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
            }
            estimator = estimator_classes[config["estimator"]](**config["params"])

            # Import train/test data from S3
            obj = s3.get_object(Bucket="data", Key="splits/healthcare-dataset-stroke-data-train.csv")
            train_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            X_train = train_df.drop(columns=["stroke"])
            y_train = train_df["stroke"]

            obj = s3.get_object(Bucket="data", Key="splits/healthcare-dataset-stroke-data-test.csv")
            test_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            X_test = test_df.drop(columns=["stroke"])
            y_test = test_df["stroke"]

            # Import the preprocessor pipeline from S3
            obj = s3.get_object(Bucket="data", Key="pipelines/preprocessor.pkl")
            preprocessor = joblib.load(io.BytesIO(obj["Body"].read()))

            # Keep preprocessing in the outer pipeline. SMOTE is fitted by the
            # inner threshold-tuning CV on each fold's training data.
            balanced_estimator = ImbPipeline(steps=[
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", estimator),
            ])

            # Tune the decision threshold together with the model hyperparameters
            # so each configuration is scored with its own optimal threshold.
            tuned_model = TunedThresholdClassifierCV(
                estimator=balanced_estimator,
                scoring="f1",
                cv=3,
                thresholds=100,
            )

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", tuned_model),
            ])
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=resolve_param_grid(config["param_grid"]),
                scoring="f1",
                cv=5,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            # Evaluate the best model; predict() applies the tuned threshold.
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "threshold": float(best_model[-1].best_threshold_),
            }

            mlflow.log_params(config["params"])
            mlflow.log_param("estimator", config["estimator"])
            mlflow.log_params({
                f"best_{key}": str(value)
                for key, value in search.best_params_.items()
            })
            mlflow.log_metric("cv_f1", search.best_score_)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                # artifact_path="model",
                serialization_format="cloudpickle",
            )

        return {
            "model": config["model"],
            "estimator": config["estimator"],
            "run_id": child_run.info.run_id,
            "model_uri": f"runs:/{child_run.info.run_id}/model",
            **metrics,
        }

    @task
    def compile_results(model_results, parent_context):
        """Select the best candidate by F1 and promote it to champion."""
        import os

        import mlflow
        from mlflow.exceptions import MlflowException

        champion_model_name = "stroke-classifier"
        champion_alias = "champion"

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        best_model = max(model_results, key=lambda x: x["f1"])

        client = mlflow.MlflowClient()
        try:
            client.get_registered_model(champion_model_name)
        except MlflowException as error:
            if "RESOURCE_DOES_NOT_EXIST" not in str(error):
                raise
            client.create_registered_model(champion_model_name)

        model_version = client.create_model_version(
            name=champion_model_name,
            source=best_model["model_uri"],
            run_id=best_model["run_id"],
            tags={"model_type": best_model["model"]},
        )
        version = int(model_version.version)

        try:
            champion = client.get_model_version_by_alias(champion_model_name, champion_alias)
        except MlflowException:
            client.set_registered_model_alias(champion_model_name, champion_alias, version)
            promoted = True
        else:
            champion_run = client.get_run(champion.run_id)
            champion_f1 = champion_run.data.metrics.get("f1")
            if champion_f1 is None:
                raise ValueError(
                    f"Champion version {champion.version} has no 'f1' run metric"
                )

            promoted = best_model["f1"] > champion_f1
            if promoted:
                client.set_registered_model_alias(champion_model_name, champion_alias, version)

        client.log_metric(parent_context["run_id"], "best_f1", best_model["f1"])
        client.set_tag(parent_context["run_id"], "selected_model", best_model["model"])

        return {
            "selected_model": best_model["model"],
            "f1": best_model["f1"],
            "version": version,
            "promote_to_champion": promoted,
        }

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def finalize_parent_success(parent_context):
        """Close the parent MLflow run as FINISHED."""
        import os

        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.MlflowClient()
        client.set_terminated(parent_context["run_id"], status="FINISHED")

        return {"run_id": parent_context["run_id"], "status": "FINISHED"}

    @task(trigger_rule=TriggerRule.ONE_FAILED)
    def finalize_parent_failure(parent_context):
        """Close the parent MLflow run as FAILED."""
        import os

        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.MlflowClient()
        client.set_terminated(parent_context["run_id"], status="FAILED")

        return {"run_id": parent_context["run_id"], "status": "FAILED"}


    preprocessor_path = create_preprocessor()

    parent_run = start_mlflow_run()

    model_results = [
        train_model.override(task_id=config["task_id"])(
            parent_run,
            preprocessor_path,
            config,
            RANDOM_STATE,
        )
        for config in MODEL_CONFIGS
    ]

    compile_result = compile_results(model_results, parent_run)

    finalize_success = finalize_parent_success(parent_run)
    finalize_failure = finalize_parent_failure(parent_run)

    compile_result >> finalize_success
    [preprocessor_path, *model_results, compile_result] >> finalize_failure


training()
