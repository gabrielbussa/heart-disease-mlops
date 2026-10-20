def test_training_pipeline_exports_artifacts_on_synthetic_fixture(synthetic_frame, tmp_path):
    from heart_mlops.data import COLUMNS, TARGET_COLUMN
    from heart_mlops.train import train

    frame = synthetic_frame.copy()
    frame = frame.rename(columns={TARGET_COLUMN: "num"})
    # Write the raw UCI schema/order expected by the acquisition loader.
    frame = frame[COLUMNS[:-1] + ["num"]]
    dataset_path = tmp_path / "synthetic-cleveland.data"
    frame.to_csv(dataset_path, index=False, header=False)
    output_dir = tmp_path / "artifacts"

    result = train(data_path=dataset_path, output_dir=output_dir)
    assert result["selected_model"] in {"logistic_regression", "random_forest"}
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "model_card.md").exists()
    assert (output_dir / "plots" / "logistic_regression_confusion_matrix.png").exists()
