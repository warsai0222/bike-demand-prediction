import logging

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from src import config

logger = logging.getLogger(__name__)


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def load_current_model():
    """Reads the registry pointer written by train.py and loads that exact
    model file -- never hardcode a filename, always go through the pointer.
    """
    with open(config.MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    model_path = config.MODELS_DIR / registry["current_model"]
    logger.info(f"Loading current model: {model_path}")
    model = joblib.load(model_path)
    return model, registry


def load_test_data():
    """Loads the processed features and slices out the frozen test window.
    Rows are NOT dropna'd here on all FEATURE_COLS globally -- we need to
    keep the naive-baseline column (cnt_lag_168h) available even for rows
    that might be missing it, so each evaluation masks what it needs.
    """
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    test_df = df[(df['timestamp'] >= config.TEST_START) & (df['timestamp'] <= config.TEST_END)].copy()
    logger.info(f"Test set: {len(test_df)} rows ({config.TEST_START.date()} -> {config.TEST_END.date()})")
    return test_df


def evaluate_model(model, test_df, feature_cols=None, target_col=None):
    """Scores the final model on the test set. Rows with any missing
    feature are dropped here (not upstream) since this is a modeling
    decision, same reasoning as load_model_ready_data() in train.py.
    """
    feature_cols = feature_cols or config.FEATURE_COLS
    target_col = target_col or config.TARGET_COL

    test_final = test_df.dropna(subset=feature_cols).reset_index(drop=True)
    preds = np.clip(model.predict(test_final[feature_cols]), 0, None)
    y_true = test_final[target_col].values

    metrics = {
        "mae": mean_absolute_error(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
        "mape": mape(y_true, preds),
    }
    logger.info(f"Final model -- Test MAE={metrics['mae']:.1f} RMSE={metrics['rmse']:.1f} MAPE={metrics['mape']:.1f}%")

    test_final = test_final.copy()
    test_final["prediction"] = preds
    test_final["residual"] = test_final[target_col] - preds
    return test_final, metrics


def evaluate_naive_baseline(test_df, target_col=None):
    """Naive seasonal baseline: predict this hour's count using the same
    hour exactly 168h (1 week) ago. Uses its own notna mask on
    cnt_lag_168h rather than FEATURE_COLS' dropna, since that column
    isn't one of the model's features.
    """
    target_col = target_col or config.TARGET_COL
    lag_col = f"{target_col}_lag_168h"

    naive_df = test_df[test_df[lag_col].notna()].copy()
    preds = naive_df[lag_col].values
    y_true = naive_df[target_col].values

    metrics = {
        "mae": mean_absolute_error(y_true, preds),
        "rmse": np.sqrt(mean_squared_error(y_true, preds)),
        "mape": mape(y_true, preds),
    }
    logger.info(f"Naive seasonal -- Test MAE={metrics['mae']:.1f} RMSE={metrics['rmse']:.1f} MAPE={metrics['mape']:.1f}%")
    return metrics


def residual_bias_check(test_final, target_col=None):
    """Mean residual: positive means the model underpredicts on average,
    negative means it overpredicts. Should be close to zero for an
    unbiased model.
    """
    target_col = target_col or config.TARGET_COL
    mean_residual = test_final["residual"].mean()
    logger.info(f"Mean residual (actual - predicted): {mean_residual:.2f}")
    return mean_residual


def save_metrics(model_metrics, naive_metrics, path=None):
    """Writes scalar test metrics to a small JSON file DVC can track as a
    'metrics' output -- this is what makes `dvc metrics diff` meaningful
    across commits (e.g. comparing this training run's test MAE against
    last week's).
    """
    path = path or (config.PROJECT_ROOT / "reports" / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_mae": model_metrics["mae"],
        "test_rmse": model_metrics["rmse"],
        "test_mape": model_metrics["mape"],
        "naive_mae": naive_metrics["mae"],
        "naive_rmse": naive_metrics["rmse"],
        "naive_mape": naive_metrics["mape"],
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {path}")


def plot_residuals(test_final, out_dir=None):
    """Saves the three diagnostic plots from the notebook: residuals vs
    predicted (heteroscedasticity check), residual distribution, and
    residuals over time.
    """
    out_dir = out_dir or (config.PROJECT_ROOT / "reports" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(test_final["prediction"], test_final["residual"], alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs Predicted")
    fig.savefig(out_dir / "residuals_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(test_final["residual"], bins=50)
    ax.set_title("Residual Distribution")
    fig.savefig(out_dir / "residual_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_final["timestamp"], test_final["residual"], alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals Over Time")
    fig.savefig(out_dir / "residuals_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved residual plots to {out_dir}")


def residual_breakdown_tables(test_final):
    """Groupby tables from the notebook: mean residual and count by hour,
    weathersit, holiday, and workingday. Surfaces where the model is
    systematically over/underpredicting.
    """
    tables = {}
    for group_col in ["hr", "weathersit", "holiday", "workingday"]:
        tables[group_col] = (
            test_final.groupby(group_col)["residual"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "mean_residual"})
        )
        logger.info(f"\nResidual by {group_col}:\n{tables[group_col]}")
    return tables


def evaluate():
    model, registry = load_current_model()
    test_df = load_test_data()

    test_final, model_metrics = evaluate_model(model, test_df)
    naive_metrics = evaluate_naive_baseline(test_df)

    residual_bias_check(test_final)
    tables = residual_breakdown_tables(test_final)
    plot_residuals(test_final)
    save_metrics(model_metrics, naive_metrics)

    return {
        "model_metrics": model_metrics,
        "naive_metrics": naive_metrics,
        "residual_tables": tables,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate()