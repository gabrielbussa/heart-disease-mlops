# Monitoring

The FastAPI service exposes Prometheus text-format metrics at `/metrics`:
- `heart_api_requests_total{method,route,status}` — request counts
- `heart_api_request_latency_seconds_sum` and `_count` — cumulative request latency and count
- `heart_api_model_loaded` — model loaded at startup (0/1)

The API also logs request method, path, HTTP status and latency, and sets `X-Process-Time-ms` on each response. The metrics are in-process and reset on restart; production deployment should use persistent monitoring and alerting.

Prometheus configuration assumes the API is reachable as `heart-disease-api:8000` from the Prometheus process. The Grafana dashboard JSON in this directory can be imported into Grafana after Prometheus is configured as a data source.
