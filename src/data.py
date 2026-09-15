import logging
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

def load_raw_data(path=None):
    """Loads the raw hourly bike-sharing CSV and runs sanity checks before
    returning it. Raises if the data doesn't look like what the rest of the
    pipeline expects -- better to fail loudly here than produce garbage
    features downstream."""

    path = path or config.RAW_DATA_PATH
    logger.info(f"Loading raw data from {path}")

    df = pd.read_csv(path, parse_dates=['dteday'])
    _validate_raw_data(df)

    logger.info("Raw data loaded successfully")
    return df

def _validate_raw_data(df):
    """ Basic schema/sanity checks not exhuastive, raises if the data doesn't meet expectations."""
    expected_cols = {
    'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
    'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
    'casual', 'registered', 'cnt'
}

    missing_cols = expected_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns:{missing_cols}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Raw data has unexpected nulls in columns: {null_cols}")

    if not(df['cnt']==df['casual']+df['registered']).all():
        raise ValueError("Raw data has inconsistent 'cnt' values: 'cnt' should be the sum of 'casual' and 'registered'")

    if (df['temp'] < 0).any() or (df['temp'] > 1).any():
        raise ValueError("Raw data has 'temp' values outside the expected range [0, 1]")

    if (df['atemp'] < 0).any() or (df['atemp'] > 1).any():
        raise ValueError("Raw data has 'atemp' values outside the expected range [0, 1]")

    if (df['hum'] < 0).any() or (df['hum'] > 1).any():
        raise ValueError("Raw data has 'hum' values outside the expected range [0, 1]")

    if (df['windspeed'] < 0).any() or (df['windspeed'] > 1).any():
        raise ValueError("Raw data has 'windspeed' values outside the expected range [0, 1]")

    if (df['cnt'] < 0).any():
        raise ValueError("Raw data has 'cnt' values less than 0")

    if not df['hr'].between(0,23).all():
        raise ValueError("Raw data has 'hr' values outside the expected range [0, 23]")

    if not df['weekday'].between(0,6).all():
        raise ValueError("Raw data has 'weekday' values outside the expected range [0, 6]")

    if not df['season'].between(1,4).all():
        raise ValueError("Raw data has 'season' values outside the expected range [1, 4]")


    logger.info("Raw data Validation passed")


def chronological_split(df,test_start=None, test_end=None):
    """Splits into dev (everything before test_start) """

    test_start = test_start or config.TEST_START
    test_end = test_end or config.TEST_END

    dev_df = df[df['dteday'] < test_start].reset_index(drop=True)
    test_df = df[(df['dteday'] >= test_start) & (df['dteday'] <= test_end)].reset_index(drop=True)

    logger.info(f"Dev set: {len(dev_df)} rows (< {test_start.date()})")
    logger.info(f"Test set: {len(test_df)} rows ({test_start.date()} -> {test_end.date()})")
    return dev_df, test_df

def get_fold(data, train_end_exclusive,val_start,val_end_inclusive,date_col='dteday'):
    """Returns a train and validation fold based on the specified date ranges."""

    train_fold = data[data[date_col]<train_end_exclusive]
    val_fold = data[(data[date_col]>= val_start) & (data[date_col]<=val_end_inclusive)]
    return train_fold,val_fold

if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    df = load_raw_data()
    dev_df, test_df = chronological_split(df)
    for i, (train_end, val_start, val_end) in enumerate(config.FOLD_BOUNDARIES):
        train_fold, val_fold = get_fold(dev_df, train_end, val_start, val_end)
        print(f"Fold {i}: train={len(train_fold)} val={len(val_fold)}")


    