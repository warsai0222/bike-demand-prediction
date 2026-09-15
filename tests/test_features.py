import pandas as pd
import pytest

from src.features import add_rush_hour_flag, build_lag_rolling_features, add_timestamp


def make_hourly_df(start, hours, workingday=1):
    """Small helper: builds a minimal hourly dataframe for feature tests."""
    dteday = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({
        "timestamp": dteday,
        "hr": dteday.hour,
        "workingday": workingday,
        "cnt": range(hours),
    })


class TestAddRushHourFlag:
    def test_morning_rush_is_inclusive_on_both_ends(self):
        """Regression test for the .isin() bug: hour 8 (the middle of the
        7-9 morning window) must be flagged, not just the endpoints 7 and 9.
        """
        df = make_hourly_df("2012-01-02", 24)  # a Monday -- a working day
        result = add_rush_hour_flag(df, morning=(7, 9), evening=(17, 19))

        for hr in [7, 8, 9]:
            assert result.loc[result["hr"] == hr, "is_rush_hour"].iloc[0] == 1, \
                f"hour {hr} should be flagged as morning rush"

    def test_evening_rush_is_inclusive_on_both_ends(self):
        df = make_hourly_df("2012-01-02", 24)
        result = add_rush_hour_flag(df, morning=(7, 9), evening=(17, 19))

        for hr in [17, 18, 19]:
            assert result.loc[result["hr"] == hr, "is_rush_hour"].iloc[0] == 1, \
                f"hour {hr} should be flagged as evening rush"

    def test_non_rush_hours_not_flagged(self):
        df = make_hourly_df("2012-01-02", 24)
        result = add_rush_hour_flag(df, morning=(7, 9), evening=(17, 19))

        for hr in [0, 6, 10, 16, 20, 23]:
            assert result.loc[result["hr"] == hr, "is_rush_hour"].iloc[0] == 0, \
                f"hour {hr} should NOT be flagged as rush hour"

    def test_rush_hour_requires_workingday(self):
        """Rush hour flag should only apply on working days -- a weekend at
        8am is not commute traffic.
        """
        df = make_hourly_df("2012-01-02", 24, workingday=0)
        result = add_rush_hour_flag(df, morning=(7, 9), evening=(17, 19))
        assert result["is_rush_hour"].sum() == 0


class TestBuildLagRollingFeatures:
    def test_lag_24h_pulls_exact_value_from_24h_ago(self):
        df = make_hourly_df("2012-01-01", 48)
        result = build_lag_rolling_features(df, target_col="cnt")

        # row at hour index 30 should have cnt_lag_24h == cnt from hour index 6
        row_30 = result[result["timestamp"] == df["timestamp"].iloc[30]].iloc[0]
        assert row_30["cnt_lag_24h"] == df["cnt"].iloc[6]

    def test_lag_is_nan_when_no_row_24h_ago(self):
        """The first 24 hours have no prior-day value to look up."""
        df = make_hourly_df("2012-01-01", 48)
        result = build_lag_rolling_features(df, target_col="cnt")

        first_row = result[result["timestamp"] == df["timestamp"].iloc[0]].iloc[0]
        assert pd.isna(first_row["cnt_lag_24h"])

    def test_lag_survives_a_gap_in_timestamps(self):
        """This is the Hurricane-Sandy scenario: if an hour is simply
        missing from the data, the timestamp-merge approach should return
        NaN for that lag, never silently pull in the wrong row by position.
        """
        df = make_hourly_df("2012-01-01", 48)
        df_with_gap = df.drop(df.index[10]).reset_index(drop=True)  # remove one hour
        result = build_lag_rolling_features(df_with_gap, target_col="cnt")

        # the row 24h after the missing hour has no valid lag source
        gap_timestamp = df["timestamp"].iloc[10] + pd.Timedelta(hours=24)
        row = result[result["timestamp"] == gap_timestamp]
        if len(row):
            assert pd.isna(row.iloc[0]["cnt_lag_24h"])

    def test_rolling_avg_excludes_current_row(self):
        """closed='left' means the current hour's own count must not leak
        into its own rolling average.
        """
        df = make_hourly_df("2012-01-01", 48)
        df["cnt"] = 100  # constant, so any leak or non-leak is easy to see
        df.loc[df.index[-1], "cnt"] = 99999  # spike on the very last row
        result = build_lag_rolling_features(df, target_col="cnt")

        last_row = result.iloc[-1]
        assert last_row["cnt_avg_prev_24h"] < 1000, \
            "rolling average leaked the current row's own value"
