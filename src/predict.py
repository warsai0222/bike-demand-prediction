import logging
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import engineer_all_features

logger = logging.getLogger(__name__)


def load_current_model():
    """Reads the registry pointer written by train.py and loads that exact
    model file. Shared by evaluate.py, monitor.py, and any future serving
    layer (Streamlit) so there's exactly one way to find "the current model."
    """
    with open(config.MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    model_path = config.MODELS_DIR / registry["current_model"]
    logger.info(f"Loading current model: {model_path}")
    model = joblib.load(model_path)
    return model, registry


def predict_on_features(model, feature_df, feature_cols=None):
    """Predicts on rows that already have engineered features (e.g. from
    the processed parquet). Rows missing any feature get a NaN prediction
    instead of raising -- callers decide how to handle those (skip them,
    log a warning, etc.) rather than this function silently dropping rows
    out from under them.
    """
    feature_cols = feature_cols or config.FEATURE_COLS

    preds = np.full(len(feature_df), np.nan)
    has_all_features = feature_df[feature_cols].notna().all(axis=1).values

    if has_all_features.any():
        valid_preds = model.predict(feature_df.loc[has_all_features, feature_cols])
        preds[has_all_features] = np.clip(valid_preds, 0, None)

    n_skipped = (~has_all_features).sum()
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} rows with missing features (returned NaN prediction)")

    return preds


def predict_new_raw(new_raw_df, historical_raw_df=None, target_col=None):
    """End-to-end prediction for genuinely new raw rows (e.g. from a
    Streamlit form or a fresh data pull) -- ones this model wasn't trained
    on and that don't have lag/rolling features computed yet.

    Lag and rolling features need history, so this concatenates the new
    rows onto existing historical raw data, re-runs the same
    engineer_all_features() pipeline used in training, and returns
    predictions for just the new rows. Loads historical_raw_df from
    config.RAW_DATA_PATH if not given.
    """
    target_col = target_col or config.TARGET_COL

    if historical_raw_df is None:
        from src.data import load_raw_data
        historical_raw_df = load_raw_data()

    combined = pd.concat([historical_raw_df, new_raw_df], ignore_index=True)
    combined = engineer_all_features(combined, target_col=target_col)

    model, _ = load_current_model()
    new_start_idx = len(historical_raw_df)
    new_features = combined.iloc[new_start_idx:].reset_index(drop=True)

    preds = predict_on_features(model, new_features)
    return preds


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()

    model, registry = load_current_model()
    print(f"Current model: {registry['current_model']}")
    print(f"Trained at: {registry['trained_at']}")
    print(f"Val metrics: {registry['val_metrics']}")
