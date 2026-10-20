import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from heart_mlops.data import FEATURE_COLUMNS, TARGET_COLUMN

@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(42)
    n = 80
    frame = pd.DataFrame({
        "age": rng.integers(30, 80, n),
        "trestbps": rng.integers(90, 180, n),
        "chol": rng.integers(150, 350, n),
        "thalach": rng.integers(90, 200, n),
        "oldpeak": rng.uniform(0, 4, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    frame[TARGET_COLUMN] = (frame["age"] + frame["oldpeak"] * 5 + frame["cp"] * 2 > 60).astype(int)
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]
