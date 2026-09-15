"""Drift monitoring via simulated replay.

This dataset is historically frozen (ends Dec 2012), so there is no live
feed of new data to watch. Instead, this module replays the frozen test
window as if it were arriving in daily batches, to demonstrate the actual
monitoring logic: track a rolling error metric, compare it against a
threshold derived from validation performance, and trigger a retrain when
it's exceeded. In production, load_stream_data() would instead pull
freshly logged predictions/actuals -- everything else here would be
unchanged.
"""
import logging
import subprocess
from collections import deque

import numpy as np
import pandas as pd

from src import config, predict

logger = logging.getLogger(__name__)


def load_stream_data():
    """Loads the frozen test window as the source of "incoming" data."""
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    test_df = df[(df["timestamp"] >= config.TEST_START) & (df["timestamp"] <= config.TEST_END)].copy()
    test_df = test_df.dropna(subset=config.FEATURE_COLS).sort_values("timestamp").reset_index(drop=True)
    return test_df


def make_batches(df, batch_size_hours=None):
    """Yields sequential chunks of `df`, batch_size_hours rows at a time --
    simulating data arriving in fixed-size chunks (e.g. one day) instead of
    all at once.
    """
    batch_size_hours = batch_size_hours or config.SIMULATED_BATCH_SIZE
    for start in range(0, len(df), batch_size_hours):
        yield df.iloc[start:start + batch_size_hours]


def log_batch_predictions(batch, preds, path=None):
    """Appends one batch's predictions/actuals/errors to the prediction
    log CSV -- the audit trail a real monitoring system builds up over time.
    """
    path = path or config.PREDICTION_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    log_rows = pd.DataFrame({
        "timestamp": batch["timestamp"].values,
        "actual": batch[config.TARGET_COL].values,
        "predicted": preds,
        "abs_error": np.abs(batch[config.TARGET_COL].values - preds),
    })
    write_header = not path.exists()
    log_rows.to_csv(path, mode="a", header=write_header, index=False)


def trigger_retrain():
    """Kicks off the DVC pipeline to retrain.

    Note: since this dataset is frozen, retraining on the same historical
    data reproduces the same model -- in a live system new data would have
    accumulated by the time drift is detected, so this call would actually
    change the model. Here it demonstrates the trigger mechanism itself.
    """
    logger.warning("Drift threshold exceeded -- triggering retrain via `dvc repro`")
    result = subprocess.run(
        ["dvc", "repro"], cwd=config.PROJECT_ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("Retrain completed successfully")
    else:
        logger.error(f"Retrain failed:\n{result.stderr}")
    return result.returncode == 0


def run_monitor(rolling_window_days=None, drift_multiplier=None, batch_size_hours=None, auto_retrain=False):
    rolling_window_days = rolling_window_days or config.ROLLING_WINDOWS
    drift_multiplier = drift_multiplier or config.DRIFT_MAE_MULTIPLIER
    batch_size_hours = batch_size_hours or config.SIMULATED_BATCH_SIZE

    model, registry = predict.load_current_model()
    baseline_mae = registry["val_metrics"]["val_mae_mean"]
    threshold = baseline_mae * drift_multiplier
    logger.info(f"Drift threshold: {threshold:.2f} ({drift_multiplier}x baseline MAE of {baseline_mae:.2f})")

    stream_df = load_stream_data()
    rolling_window_hours = rolling_window_days * 24
    error_history = deque()  # (timestamp, abs_error) pairs within the rolling window

    already_triggered = False

    for batch in make_batches(stream_df, batch_size_hours):
        preds = predict.predict_on_features(model, batch)
        abs_errors = np.abs(batch[config.TARGET_COL].values - preds)

        log_batch_predictions(batch, preds)

        for ts, err in zip(batch["timestamp"], abs_errors):
            error_history.append((ts, err))

        cutoff = batch["timestamp"].max() - pd.Timedelta(hours=rolling_window_hours)
        while error_history and error_history[0][0] < cutoff:
            error_history.popleft()

        rolling_mae = float(np.mean([err for _, err in error_history]))
        batch_end = batch["timestamp"].max()
        logger.info(
            f"[{batch_end}] batch_mae={abs_errors.mean():.2f} "
            f"rolling_mae={rolling_mae:.2f} (threshold={threshold:.2f})"
        )

        if rolling_mae > threshold and not already_triggered:
            logger.warning(f"[{batch_end}] Rolling MAE {rolling_mae:.2f} exceeded threshold {threshold:.2f}")
            if auto_retrain:
                trigger_retrain()
                already_triggered = True  # avoid retriggering every subsequent batch this run
            else:
                logger.warning("auto_retrain=False -- would trigger retrain here in production")

    logger.info("Monitoring simulation complete")


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    run_monitor(auto_retrain=False)
