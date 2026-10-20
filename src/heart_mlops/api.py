"""FastAPI inference service. Requires artifacts/model.joblib produced by training."""
import os
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0",
              description="Educational classifier; not for medical use.")
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

# Small in-process Prometheus metrics. These reset when the process restarts;
# use a metrics backend and durable monitoring for production deployments.
REQUEST_COUNT = {}
REQUEST_LATENCY_SUM = 0.0
REQUEST_LATENCY_COUNT = 0

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-5, le=15)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)

class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["negative", "positive"]
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str

@app.middleware("http")
async def log_requests(request, call_next):
    import logging, time
    logger = logging.getLogger("heart_disease_api")
    started = time.perf_counter()
    global REQUEST_LATENCY_SUM, REQUEST_LATENCY_COUNT
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    route = request.url.path
    key = (request.method, route, response.status_code)
    REQUEST_COUNT[key] = REQUEST_COUNT.get(key, 0) + 1
    REQUEST_LATENCY_SUM += elapsed_ms / 1000.0
    REQUEST_LATENCY_COUNT += 1
    logger.info("method=%s path=%s status=%s latency_ms=%.2f",
                request.method, route, response.status_code, elapsed_ms)
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose Prometheus text exposition format for request count and latency."""
    lines = [
        "# HELP heart_api_requests_total Total HTTP requests handled by this process.",
        "# TYPE heart_api_requests_total counter",
    ]
    for (method, route, status), count in sorted(REQUEST_COUNT.items()):
        safe_route = route.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        lines.append(f'heart_api_requests_total{{method="{method}",route="{safe_route}",status="{status}"}} {count}')
    lines.extend([
        "# HELP heart_api_request_latency_seconds_sum Total observed request latency in seconds.",
        "# TYPE heart_api_request_latency_seconds_sum counter",
        f"heart_api_request_latency_seconds_sum {REQUEST_LATENCY_SUM}",
        "# HELP heart_api_request_latency_seconds_count Number of requests included in latency total.",
        "# TYPE heart_api_request_latency_seconds_count counter",
        f"heart_api_request_latency_seconds_count {REQUEST_LATENCY_COUNT}",
        "# HELP heart_api_model_loaded Whether a model artifact was loaded at startup.",
        "# TYPE heart_api_model_loaded gauge",
        f"heart_api_model_loaded {1 if model is not None else 0}",
    ])
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8")

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
    frame = pd.DataFrame([payload.model_dump()])
    try:
        probabilities = model.predict_proba(frame)[0]
        prediction = int(model.predict(frame)[0])
        probs = {str(i): float(p) for i, p in enumerate(probabilities)}
        return PredictionResponse(
            prediction=prediction,
            label="positive" if prediction == 1 else "negative",
            confidence=float(max(probabilities)),
            probabilities=probs,
            disclaimer="Educational demonstration only; not a medical diagnosis."
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
