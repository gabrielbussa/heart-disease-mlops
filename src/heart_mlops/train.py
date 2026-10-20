"""Train, cross-validate, compare, track, and export two binary classifiers."""
from pathlib import Path
import json
import warnings
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from .data import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN, validate_dataset
from .pipeline import make_model_pipeline

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}


def train(data_path="data/raw/processed.cleveland.data", output_dir="artifacts"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    validate_dataset(df)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    if y.nunique() < 2 or y.value_counts().min() < 5:
        raise ValueError("Need at least two target classes and enough samples for stratified 5-fold CV.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name, best_score, best_pipeline = None, -1.0, None
    mlflow = None
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_experiment("uci-heart-disease-assignment-01")
    except ImportError:
        warnings.warn("MLflow is not installed; continuing without experiment tracking.", RuntimeWarning)

    for name in ["logistic_regression", "random_forest"]:
        pipeline = make_model_pipeline(name)
        if name == "logistic_regression":
            param_grid = {"classifier__C": [0.1, 1.0, 10.0]}
        else:
            param_grid = {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 8],
                "classifier__min_samples_leaf": [1, 2],
            }
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        best_idx = search.best_index_
        metrics = {
            "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][best_idx]),
            "cv_precision_mean": float(search.cv_results_["mean_test_precision"][best_idx]),
            "cv_recall_mean": float(search.cv_results_["mean_test_recall"][best_idx]),
            "cv_roc_auc_mean": float(search.best_score_),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        results[name] = {
            "best_params": search.best_params_,
            "tuning": {
                "method": "GridSearchCV with stratified 5-fold CV",
                "selection_metric": "ROC-AUC",
                "candidate_parameter_combinations": int(len(search.cv_results_["params"])),
            },
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        }
        # Generate a genuine confusion matrix plot for every trained candidate.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        plot_dir = out / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{name}_confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["negative", "positive"])
        plt.title(f"{name.replace('_', ' ').title()} — Holdout Confusion Matrix")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "random_state": 42, "test_size": 0.20, "cv_folds": 5, **search.best_params_})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
                mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="pickle")
        # Select using CV ROC-AUC only; the holdout set is not used for model selection.
        if metrics["cv_roc_auc_mean"] > best_score:
            best_score, best_name, best_pipeline = metrics["cv_roc_auc_mean"], name, pipeline

    joblib.dump(best_pipeline, out / "model.joblib")
    results["selected_model"] = best_name
    results["selection_criterion"] = "Highest mean 5-fold CV ROC-AUC after GridSearchCV tuning on training split; holdout test set used only for final evaluation"
    results["dataset_rows"] = int(len(df))
    results["train_rows"] = int(len(X_train))
    results["test_rows"] = int(len(X_test))
    results["positive_label_definition"] = "num > 0 maps to target=1; num == 0 maps to target=0"
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    model_card = f"""# Model Card: UCI Heart Disease Educational Classifier

- Dataset: UCI Heart Disease Cleveland processed subset.
- Target: binary; original `num=0` is negative, `num=1..4` is positive.
- Selected model: {best_name}
- Selection criterion: highest mean 5-fold cross-validation ROC-AUC after model-specific GridSearchCV tuning on the training split.
- Train/test split: stratified 80/20, random_state=42.
- Intended use: educational demonstration of MLOps workflows only.
- Not intended use: medical diagnosis, treatment, triage, or clinical decision-making.
- Limitations: small historical dataset, potential sampling bias, no external clinical validation, and confidence scores are not necessarily calibrated.
"""
    (out / "model_card.md").write_text(model_card)
    return results


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
