from sklearn.model_selection import train_test_split
from heart_mlops.data import FEATURE_COLUMNS, TARGET_COLUMN
from heart_mlops.pipeline import make_model_pipeline

def test_logistic_pipeline_fits_and_predicts(synthetic_frame):
    X = synthetic_frame[FEATURE_COLUMNS]
    y = synthetic_frame[TARGET_COLUMN]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = make_model_pipeline("logistic_regression")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

def test_unknown_model_raises():
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        make_model_pipeline("not-a-model")
