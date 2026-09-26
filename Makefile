include .env

AIRFLOW_PORT ?= 8080
MLFLOW_PORT ?= 5001
MINIO_PORT_UI ?= 9001
FASTAPI_PORT ?= 8800

.PHONY: help start urls stop clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  start   Start all services (detached) and print UI URLs"
	@echo "  urls    Print available web UI URLs"
	@echo "  stop    Stop all services"
	@echo "  clean   Stop services and remove images and volumes"

start:
	docker compose --profile all up -d
	@$(MAKE) urls

urls:
	@echo "Apache Airflow: http://localhost:$(AIRFLOW_PORT)"
	@echo "MLflow:         http://localhost:$(MLFLOW_PORT)"
	@echo "MinIO console:  http://localhost:$(MINIO_PORT_UI)"
	@echo "FastAPI:        http://localhost:$(FASTAPI_PORT)"
	@echo "FastAPI docs:   http://localhost:$(FASTAPI_PORT)/docs"

stop:
	docker compose --profile all down

clean:
	docker compose down --rmi all --volumes
