from fastapi.testclient import TestClient
import heart_mlops.api as api

def test_health_endpoint_without_artifact(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False

def test_prediction_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    client = TestClient(api.app)
    payload = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503

def test_prediction_rejects_invalid_input():
    client = TestClient(api.app)
    response = client.post("/predict", json={"age": -3})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    client = TestClient(api.app)
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "heart_api_requests_total" in response.text
    assert "heart_api_model_loaded 0" in response.text


def test_prediction_success_with_fitted_pipeline(monkeypatch, synthetic_frame):
    from heart_mlops.data import FEATURE_COLUMNS, TARGET_COLUMN
    from heart_mlops.pipeline import make_model_pipeline
    model = make_model_pipeline("logistic_regression")
    model.fit(synthetic_frame[FEATURE_COLUMNS], synthetic_frame[TARGET_COLUMN])
    monkeypatch.setattr(api, "model", model)
    client = TestClient(api.app)
    payload = {
        "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
        "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["confidence"] <= 1
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
