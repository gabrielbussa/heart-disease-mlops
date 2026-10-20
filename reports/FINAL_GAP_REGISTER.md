# Final gap register — assignment rubric mapped

This register separates implemented artifacts from external actions that cannot be truthfully claimed without a reachable dataset host, a GitHub account/repository, and a Docker/cloud runtime.

| Rubric item | Prepared in bundle | Current verification | Remaining evidence |
|---|---|---|---|
| 1. Data acquisition & EDA (5) | UCI download script, validation, EDA JSON and plot generation | Official Cleveland dataset processed in Colab: 303 rows; EDA outputs generated | EDA summary and plots generated in reports/figures/ |
| 2. Feature engineering & models (8) | Imputation, scaling, one-hot encoding, Logistic Regression and Random Forest; GridSearchCV and 5-fold stratified CV | Both models trained on official Cleveland data; holdout metrics saved in artifacts/metrics.json | Run training on official dataset and review real metrics |
| 3. Experiment tracking (5) | MLflow experiment runs, params, metrics, confusion-matrix plots and model artifacts | MLflow not installed here; no actual MLflow run | MLflow runs logged locally in Colab; hosted/shared tracking server and screenshots not verified |
| 4. Packaging & reproducibility (7) | joblib pipeline export, requirements, model card, notebook and setup docs | Source compile passed; no official model artifact | Verify clean environment training and artifact reload |
| 5. CI/CD & testing (8) | Ruff/Pytest/real-data training/Docker build/smoke-test GitHub Actions workflow | 12 tests passed in the Colab run after the MLflow serialization adjustment; hosted GitHub Actions workflow remains unverified | Push to repository and inspect successful workflow logs/artifacts |
| 6. Containerization (5) | Dockerfile, FastAPI `/predict`, `/health`, `/metrics`; CI smoke-test commands | Docker engine unavailable here | Build image and test actual container |
| 7. Production deployment (7) | Kubernetes Deployment and LoadBalancer Service manifest | Not deployed; image registry placeholder must be replaced | Publish image, deploy to cloud or local Kubernetes, capture endpoint/screenshots |
| 8. Monitoring & logging (3) | Request logs, latency header, Prometheus text metrics, scrape config, Grafana dashboard JSON | API metrics endpoint covered by unit tests; no live Prometheus/Grafana stack | Run stack and capture dashboard with real traffic |
| 9. Documentation/reporting (2) | 10-page DOCX draft, README, architecture PNG/Mermaid, video script | Files generated; report intentionally labels unverified execution | Add actual metrics, screenshots, repository URL, endpoint details and video |

## Environment limitations encountered
- The runtime cannot resolve the official UCI host (DNS/network restriction).
- Docker engine is not available in the build runtime.
- MLflow and Ruff are not installed in the build runtime; the CI workflow installs declared dependencies on its hosted runner.
- No connected GitHub account/repository, cloud credentials, Kubernetes cluster, or screen-recording device session is available to this build process.

No real dataset metrics, successful cloud deployment, public API URL, screenshots, or video are fabricated. These are the remaining non-automatable items without access to external services.
