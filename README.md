# Heart Disease MLOps — BITS Pilani WILP AIMLCZG523 Assignment 01

This repository is structured against the official 50-mark assignment brief. It includes dataset acquisition, EDA, two-model training, cross-validation, optional MLflow tracking, a reusable preprocessing/model pipeline, a FastAPI prediction API, Pytest tests, GitHub Actions CI, Docker packaging, Kubernetes manifests, monitoring configuration, and report/video guidance.

> **Honest execution status:** The source code and submission structure have been generated. In this environment the official UCI dataset could not be downloaded, and MLflow/Docker/cloud credentials are unavailable. Therefore, real-dataset model metrics, remote CI execution, container build, deployment, screenshots, and video are **not claimed as completed**. The tests use synthetic data only to validate software behavior; they do not represent model performance.

## Dataset

Use the Cleveland subset of the UCI Heart Disease dataset (303 instances, 13 predictors; target field `num`). Official source: https://archive.ics.uci.edu/dataset/45/heart+disease

The raw processed Cleveland file has 14 columns without a header. Missing values are represented by `?`. This project maps `num=0` to target 0 and `num=1..4` to target 1. This is a binary educational classification exercise, not a clinical diagnostic tool.

## Project layout

- `src/heart_mlops/data.py` — acquisition and data preparation
- `src/heart_mlops/eda.py` — EDA summaries and plots
- `src/heart_mlops/pipeline.py` — reproducible preprocessing/model pipeline
- `src/heart_mlops/train.py` — cross-validation, holdout evaluation, MLflow logging, model export
- `src/heart_mlops/api.py` — FastAPI `/predict` endpoint
- `tests/` — automated tests using synthetic fixtures
- `notebooks/01_eda_and_modeling.ipynb` — guided notebook
- `.github/workflows/ci.yml` — lint, tests, dataset-backed training job
- `Dockerfile` — API container
- `deployment/k8s.yaml` — Kubernetes Deployment and Service
- `monitoring/` — Prometheus scrape configuration
- `reports/` — report draft and video narration
- `screenshots/README.md` — evidence checklist

## Local or cloud setup

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
python -m heart_mlops.eda
python -m heart_mlops.train
uvicorn heart_mlops.api:app --host 0.0.0.0 --port 8000
```

Set the package path before running modules from the repository root:

```bash
export PYTHONPATH=src
# Windows PowerShell: $env:PYTHONPATH = "src"
```

The training script creates `artifacts/model.joblib`, `artifacts/metrics.json`, and `artifacts/model_card.md`. MLflow tracking is enabled if MLflow is installed; set `MLFLOW_TRACKING_URI` to use a remote tracking server.

## API example

`POST /predict`:

```json
{
  "age": 57,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 236,
  "fbs": 0,
  "restecg": 0,
  "thalach": 174,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

Response includes `prediction`, `confidence`, and class probabilities. Confidence is a model score, not a medically calibrated risk estimate.

## Docker

After running training and creating `artifacts/model.joblib`:

```bash
docker build -t heart-disease-mlops:local .
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts:ro" heart-disease-mlops:local
```

Then test `GET /health` and `POST /predict` at `http://localhost:8000`. A Docker build must be run in a Docker-capable environment; it has not been verified in the current environment.

## MLflow

Run training with MLflow installed. The default tracking URI is a local `mlruns` directory. To view the UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

For a remote/cloud setup, configure a persistent tracking server and `MLFLOW_TRACKING_URI`. Do not assume local tracking files are automatically available in a hosted runtime after it shuts down.

## GitHub Actions

The workflow runs linting, unit tests, a real-dataset training step, a Docker build, and container endpoint smoke tests. The training job downloads the official dataset, so it needs network access. The workflow itself must be pushed to a GitHub repository and executed by GitHub Actions before it can be represented as successful evidence.

## Kubernetes

Update the image field in `deployment/k8s.yaml` to the image you built and pushed to a registry. Apply with:

```bash
kubectl apply -f deployment/k8s.yaml
kubectl get deployments,services
kubectl port-forward service/heart-disease-api 8000:80
```

The included LoadBalancer service is cloud-dependent; on Minikube, use `minikube tunnel` or change the service to NodePort for local testing. Kubernetes deployment has not been executed here.

## Important limitations

- The dataset is small and historical; results are educational only.
- This model is not validated for medical decisions.
- The project does not contain fabricated real metrics or screenshots.
- `reports/final_report_draft.docx` is a report draft; insert actual executed metrics, screenshots, repository URL, deployment URL, and video evidence before final submission.
