from pathlib import Path
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent # Root directory of the project
PARAMS_PATH = PROJECT_ROOT / "params.yaml" # tells where the params.yaml file is located

with open(PARAMS_PATH) as f:
    _params = yaml.safe_load(f) #_params now holds the contents of params.yaml as a dictionary

#--Paths (resolved to absolute paths realtive to project root)--

RAW_DATA_PATH = PROJECT_ROOT / _params["paths"]["raw_data"]
PROCESSED_DATA_PATH = PROJECT_ROOT / _params["paths"]["processed_data"]
MODELS_DIR = PROJECT_ROOT / _params["paths"]["models_dir"]
MODEL_REGISTRY_PATH = PROJECT_ROOT / _params["paths"]["model_registry"]
PREDICTION_LOG_PATH = PROJECT_ROOT / _params["paths"]["prediction_log"]
LOG_FILE_PATH = PROJECT_ROOT / _params["paths"]["log_file"]

#Chornological split
TEST_START= pd.Timestamp(_params["split"]["test_start"])
TEST_END = pd.Timestamp(_params["split"]["test_end"])

#WALK Forward folds: List 

FOLD_BOUNDARIES = [
    (pd.Timestamp(train_end), pd.Timestamp(val_start), pd.Timestamp(val_end))
    for train_end, val_start, val_end in _params["split"]["folds"]
]

#Features
FEATURE_COLS = _params["features"]["feature_cols"]
CATEGORICAL_COLS = _params["features"]["categorical_features"]
TRAIL_AVG_WINDOW_WEEKS = _params["features"]["trail_avg_window_weeks"]
MORNING_RUSH = tuple(_params["features"]["rush_hour"]["morning"])
EVENING_RUSH = tuple(_params["features"]["rush_hour"]["evening"])

#models
TARGET_COL = _params["model"]["target_col"]
FINAL_PARAMS = _params["model"]["params"]

#MLFLOW
MLFLOW_EXPERIMENT_NAME = _params["mlflow"]["experiment_name"]
MLFLOW_TRACKING_URI = _params["mlflow"]["tracking_uri"]

#Monitoring and drift
DRIFT_MAE_MULTIPLIER = _params["monitoring"]["drift_mae"]
ROLLING_WINDOWS = _params["monitoring"]["rolling_window_days"]
SIMULATED_BATCH_SIZE = _params["monitoring"]["simulated_batch_size_hours"]

if __name__ == "__main__":
    # Quick sanity check when run directly: python src/config.py
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data path: {RAW_DATA_PATH}  (exists: {RAW_DATA_PATH.exists()})")
    print(f"Test window: {TEST_START.date()} -> {TEST_END.date()}")
    print(f"Folds: {len(FOLD_BOUNDARIES)}")
    for i, (train_end, val_start, val_end) in enumerate(FOLD_BOUNDARIES):
        print(f"  Fold {i}: train < {train_end.date()}, val {val_start.date()} -> {val_end.date()}")
    print(f"Feature count: {len(FEATURE_COLS)}")
    print(f"Categorical features: {CATEGORICAL_COLS}")
    print(f"Model params: {FINAL_PARAMS}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Drift MAE multiplier: {DRIFT_MAE_MULTIPLIER}")