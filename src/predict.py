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

    # engineer_all_features() sorts everything by timestamp internally
    # (build_lag_rolling_features), so the new row does NOT stay at the
    # tail of the dataframe unless its timestamp is chronologically after
    # every historical row. Selecting "the last N rows by position" broke
    # silently for any backtest date -- it grabbed whatever real historical
    # row happened to land at that position after sorting, which is why
    # predictions looked constant regardless of the input. A boolean
    # marker column survives the sort/merge pipeline and identifies the
    # new rows correctly no matter where they end up.
    new_keys = set(zip(
        pd.to_datetime(new_raw_df["dteday"]).dt.normalize(), new_raw_df["hr"],
    ))
    hist_keys = list(zip(
        pd.to_datetime(historical_raw_df["dteday"]).dt.normalize(), historical_raw_df["hr"],
    ))
    # If the requested (date, hour) already has a real historical row
    # (backtesting an hour that actually happened), drop that historical
    # row before concatenating. Otherwise the lag-feature merge in
    # build_lag_rolling_features (a left join on timestamp) hits a
    # duplicate key and multiplies rows -- silently corrupting the result
    # rather than raising an error.
    keep_mask = [k not in new_keys for k in hist_keys]
    n_dropped = len(keep_mask) - sum(keep_mask)
    if n_dropped:
        logger.info(f"Backtest overlaps {n_dropped} existing historical row(s); using the requested input instead")
    historical_filtered = historical_raw_df.loc[keep_mask].copy()

    historical_filtered["_is_new_row"] = False
    new_marked = new_raw_df.copy()
    new_marked["_is_new_row"] = True

    combined = pd.concat([historical_filtered, new_marked], ignore_index=True)
    combined = engineer_all_features(combined, target_col=target_col)

    model, _ = load_current_model()
    new_features = combined.loc[combined["_is_new_row"]].drop(columns=["_is_new_row"]).reset_index(drop=True)

    preds = predict_on_features(model, new_features)
    return preds


if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()

    model, registry = load_current_model()
    print(f"Current model: {registry['current_model']}")
    print(f"Trained at: {registry['trained_at']}")
    print(f"Val metrics: {registry['val_metrics']}")
