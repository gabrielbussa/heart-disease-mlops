"""Dataset acquisition and validation for the UCI Cleveland Heart Disease data."""
from pathlib import Path
from io import StringIO
import pandas as pd
import requests

DATA_URLS = [
    # Primary: official UCI processed Cleveland subset.
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    # Fallback mirror of the same UCI file for environments where the legacy UCI host is unavailable.
    "https://raw.githubusercontent.com/meghakkulal/heart-disease-prediction/main/processed.cleveland.data",
]
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "target"
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def download_dataset(destination: str | Path = "data/raw/processed.cleveland.data") -> Path:
    """Download the official processed Cleveland file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in DATA_URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("Downloaded dataset is empty.")
            # Validate the raw file shape before writing it to the expected path.
            candidate = pd.read_csv(StringIO(response.text), header=None, na_values="?")
            if candidate.shape[1] != len(COLUMNS) or len(candidate) < 250:
                raise ValueError(f"Unexpected dataset shape: {candidate.shape}")
            destination.write_bytes(response.content)
            return destination
        except Exception as exc:
            failures.append(f"{url}: {exc}")
    raise RuntimeError("Could not download a valid Cleveland dataset. Attempts: " + " | ".join(failures))


def load_dataset(path: str | Path = "data/raw/processed.cleveland.data") -> pd.DataFrame:
    """Load raw UCI file, normalize missing values, and create binary target."""
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values="?", skipinitialspace=True)
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    for col in COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["num"]).copy()
    frame[TARGET_COLUMN] = (frame["num"] > 0).astype(int)
    frame = frame.drop(columns=["num"])
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise a helpful error if required columns or target values are invalid."""
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    values = set(frame[TARGET_COLUMN].dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Binary target must contain only 0/1; found {values}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
