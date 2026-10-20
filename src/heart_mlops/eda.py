"""Exploratory analysis summary and visualizations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .data import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN, validate_dataset


def run_eda(data_path="data/raw/processed.cleveland.data", output_dir="reports/figures"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    validate_dataset(df)
    summary = {
        "rows": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "target_counts": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()},
        "descriptive_statistics": json.loads(df[FEATURE_COLUMNS].describe().to_json()),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    sns.set_theme(style="whitegrid")
    df[TARGET_COLUMN].map({0: "No disease", 1: "Disease"}).value_counts().plot(kind="bar")
    plt.title("Target Class Balance")
    plt.xlabel("Target class")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=160)
    plt.close()
    df[FEATURE_COLUMNS].hist(figsize=(14, 10), bins=15)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(out / "feature_histograms.png", dpi=160)
    plt.close()
    numeric = df[FEATURE_COLUMNS].copy()
    for c in numeric.columns:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close()
    return summary


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))
