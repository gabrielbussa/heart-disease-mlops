"""Reusable preprocessing and classification pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def make_model_pipeline(model_name: str, random_state: int = 42):
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", estimator)])
