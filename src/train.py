import logging
from datetime import datetime,timezone #timezone is used to save the model version
import json
import mlflow.lightgbm
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src import config

logger = logging.getLogger(__name__)

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)*100

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def load_model_ready_data():
    """Load and return the model-ready data."""
    df= pd.read_parquet(config.PROCESSED_DATA_PATH)
    before=len(df)
    df = df.dropna(subset=config.FEATURE_COLS).reset_index(drop=True)
    logger.info(f"Dropped {before-len(df)} rows with missing values. Remaining {len(df)} rows.")
    return df


def run_walk_forward(df,target_col=None, feature_cols=None, categorical_features=None):
    """Run a walk-forward validation on the given DataFrame."""
    target_col = target_col or config.TARGET_COL
    feature_cols = feature_cols or config.FEATURE_COLS
    categorical_features = categorical_features or config.CATEGORICAL_COLS

    fold_metrics = []

    for fold_idx,(train_end,val_start,val_end) in enumerate(config.FOLD_BOUNDARIES):
        train_fold = df[df['timestamp'] < train_end]
        val_fold = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)]

        model=lgb.LGBMRegressor(**config.FINAL_PARAMS)
        model.fit(
            train_fold[feature_cols],train_fold[target_col],
            categorical_feature=categorical_features
        )

        preds = np.clip(model.predict(val_fold[feature_cols]), 0, None)
        y_val = val_fold[target_col].values

        fold_mae = mean_absolute_error(y_val, preds)
        fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_mape = mape(y_val, preds)

        fold_metrics.append({
            "fold": fold_idx,
            "mae": fold_mae,
            "rmse": fold_rmse,
            "mape": fold_mape
        })

        logger.info(f"Fold {fold_idx} - MAE: {fold_mae:.4f}, RMSE: {fold_rmse:.4f}, MAPE: {fold_mape:.2f}%")

    return fold_metrics

def summarize_fold_metrics(fold_metrics):

    summary={}
    for metric in ['mae','rmse','mape']:
        values =[m[metric] for m in fold_metrics]
        summary[f"val_{metric}_mean"]=float(np.mean(values))
        summary[f"val_{metric}_std"]=float(np.std(values))
    return summary

def fit_final_model(df,target_col=None, feature_cols=None, categorical_features=None):
    target_col = target_col or config.TARGET_COL
    feature_cols = feature_cols or config.FEATURE_COLS
    categorical_features = categorical_features or config.CATEGORICAL_COLS


    train_final = df[df['timestamp'] < config.TEST_START]
    logger.info(
        f"Training final model on {len(train_final)} rows ({len(feature_cols)} features, (< {config.TEST_START.date()}))"
    )

    model = lgb.LGBMRegressor(**config.FINAL_PARAMS)
    model.fit(
        train_final[feature_cols], train_final[target_col],
        categorical_feature=categorical_features
    )

    return model

def save_versioned_model(model,val_summary):
    """Save the trained model with a versioned filename and updated the model
    registry pointer
    """

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_filename = f"{config.TARGET_COL}_model_{timestamp}.pkl"
    model_path = config.MODELS_DIR / model_filename

    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    registry = {
        "current_model": model_filename,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_metrics": val_summary
    }

    config.MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Updated model registry at {config.MODEL_REGISTRY_PATH}->{model_filename}")

    return model_path

def train():
    df = load_model_ready_data()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="cnt_direct_train"):
        mlflow.log_params(config.FINAL_PARAMS)
        mlflow.log_param("feature_cols", config.FEATURE_COLS)
        mlflow.log_param("categorical_features", config.CATEGORICAL_COLS)
        mlflow.log_param("target_col", config.TARGET_COL)

        fold_metrics = run_walk_forward(df)
        for fold_idx, m in enumerate(fold_metrics):
            mlflow.log_metrics({
                f"fold_{fold_idx}_mae": m["mae"],
                f"fold_{fold_idx}_rmse": m["rmse"],
                f"fold_{fold_idx}_mape": m["mape"],
            })

        val_summary = summarize_fold_metrics(fold_metrics)
        mlflow.log_metrics(val_summary)
        logger.info(f"Val summary: {val_summary}")

        model = fit_final_model(df)
        mlflow.lightgbm.log_model(model, "model")

        model_path = save_versioned_model(model, val_summary)
        mlflow.log_param("model_path", str(model_path))

    return model, val_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
    






