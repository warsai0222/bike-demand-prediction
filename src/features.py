import logging
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

def add_trend_feature(df):
    """Adds a trend feature to the dataframe based on the datetime column 'dteday'."""
    df=df.copy()
    start = df['dteday'].min()
    df['days_since_start'] = (df['dteday'] - start).dt.days
    return df

def add_timestamp(df):
    """Adds a timestamp feature to the dataframe based on the datetime column 'dteday'."""
    df = df.copy()
    df['timestamp'] = df['dteday']+pd.to_timedelta(df['hr'], unit='h') #this adds the hour component to the timestamp
    return df

def add_rush_hour_flag(df,morning=None,evening=None):
    """Adds a rush hour flag to the dataframe based on the hour column 'hr'."""
    morning = morning or config.MORNING_RUSH
    evening = evening or config.EVENING_RUSH

    df=df.copy()
    is_morning = df['hr'].between(morning[0], morning[1])
    is_evening = df['hr'].between(evening[0], evening[1])
    df['is_rush_hour'] = ((is_morning | is_evening) & (df['workingday'] == 1)).astype(int)
    return df

def build_lag_rolling_features(data,target_col,trail_avg_window=None):
    """Leakage Sage lag and rolling features for the target column."""

    trail_avg_window_weeks = trail_avg_window or config.TRAIL_AVG_WINDOW_WEEKS
    data = data.sort_values('timestamp').reset_index(drop=True)

    #exact lag via timestamp merge 
    lookup = data[['timestamp',target_col]].copy() #lookup table for rolling and lag features
    for lag_hours,name in [(24,'24h'),(168,'168h')]:
        shifted = lookup.copy()
        shifted['timestamp'] = shifted['timestamp'] + pd.to_timedelta(lag_hours, unit='h')
        shifted = shifted.rename(columns={target_col:f'{target_col}_lag_{name}'})
        data = data.merge(shifted, on='timestamp', how='left')

    #rolling features, current row excluded
    ts_indexed = data.set_index('timestamp')[target_col]
    data[f'{target_col}_avg_prev_24h']= ts_indexed.rolling('24h',closed='left').mean().values
    data[f'{target_col}_avg_prev_7d']= ts_indexed.rolling('7D',closed='left').mean().values


    #same weekday+hour trailing average (occurnece-based, self-corrects for missing dates)
    trail_col = f'{target_col}_trail_avg_hr_wd_{trail_avg_window_weeks}w'
    data[trail_col] = (data.groupby([data['timestamp'].dt.hour, data['timestamp'].dt.weekday])[target_col]\
        .transform(lambda x: x.shift(1).rolling(window=trail_avg_window_weeks, min_periods=1).mean()))

    logger.info(f"Built lag/rolling features for {target_col}")
    return data

def engineer_all_features(df,target_col= 'cnt'):
    """Runs the full feature engineering pipeline in the correct order:
    timestamp -> trend ->lag/rolling ->rush-hour flag. Single entrypoint train.py and predict.py
    should call, so feature order/logic is preserved.
    """

    df = add_timestamp(df)
    df = add_trend_feature(df)
    df = build_lag_rolling_features(df, target_col)
    df = add_rush_hour_flag(df)
    return df

def save_processed_features(df, path=None):
    """Saves the engineered feature dataframe to disk so downstream stages
    (train.py, evaluate.py) don't need to recompute features from raw data
    every time -- and so DVC can track this as a versioned pipeline output."""
    path = path or config.PROCESSED_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed features to {path} ({len(df)} rows, {len(df.columns)} columns)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data import load_raw_data

    df = load_raw_data()
    df = engineer_all_features(df, target_col='cnt')

    print(df['is_rush_hour'].value_counts())
    print(f"Rush hour rate: {df['is_rush_hour'].mean()*100:.1f}%")
    save_processed_features(df)

    print(df.columns.tolist())

    print(f"\nColumns after feature engineering: {len(df.columns)}")
    print(f"Missing values per feature (FEATURE_COLS only):")
    print(df[config.FEATURE_COLS].isna().sum()[lambda s: s > 0])

    missing_mask = df[config.FEATURE_COLS].isna().any(axis=1)
    print(f"\nRows with any missing feature: {missing_mask.sum()} out of {len(df)}")