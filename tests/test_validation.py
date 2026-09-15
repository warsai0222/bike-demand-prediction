import pandas as pd
import pytest

from src.data import _validate_raw_data, chronological_split, get_fold


def make_valid_raw_df(n=48):
    dteday = pd.date_range("2012-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "dteday": dteday.normalize(),
        "season": 1, "yr": 0, "mnth": 1, "hr": dteday.hour,
        "holiday": 0, "weekday": dteday.dayofweek, "workingday": 1,
        "weathersit": 1, "temp": 0.5, "atemp": 0.5, "hum": 0.5, "windspeed": 0.2,
        "casual": 5, "registered": 10, "cnt": 15,
    })


class TestValidateRawData:
    def test_valid_data_passes(self):
        df = make_valid_raw_df()
        _validate_raw_data(df)  # should not raise

    def test_missing_column_raises(self):
        df = make_valid_raw_df().drop(columns=["weathersit"])
        with pytest.raises(ValueError, match="missing expected columns"):
            _validate_raw_data(df)

    def test_null_values_raise(self):
        df = make_valid_raw_df()
        df.loc[0, "temp"] = None
        with pytest.raises(ValueError, match="unexpected nulls"):
            _validate_raw_data(df)

    def test_cnt_not_equal_casual_plus_registered_raises(self):
        df = make_valid_raw_df()
        df.loc[0, "cnt"] = 999  # no longer casual + registered
        with pytest.raises(ValueError, match="inconsistent 'cnt'"):
            _validate_raw_data(df)

    def test_temp_out_of_range_raises(self):
        df = make_valid_raw_df()
        df.loc[0, "temp"] = 1.5
        with pytest.raises(ValueError, match="'temp' values outside"):
            _validate_raw_data(df)

    def test_negative_cnt_raises(self):
        df = make_valid_raw_df()
        df.loc[0, "cnt"] = -1
        df.loc[0, "casual"] = -5  # keep cnt == casual + registered so that check doesn't fire first
        df.loc[0, "registered"] = 4
        with pytest.raises(ValueError, match="'cnt' values less than 0"):
            _validate_raw_data(df)


class TestChronologicalSplit:
    def test_split_is_strictly_by_date_boundary(self):
        df = make_valid_raw_df(n=72)  # 3 days
        test_start = pd.Timestamp("2012-01-02")
        test_end = pd.Timestamp("2012-01-02 23:00:00")

        dev_df, test_df = chronological_split(df, test_start=test_start, test_end=test_end)

        assert (dev_df["dteday"] < test_start).all()
        assert (test_df["dteday"] >= test_start).all()
        assert (test_df["dteday"] <= test_end).all()
        # no row should be lost or duplicated between dev/test for this window
        assert len(dev_df) + len(test_df) <= len(df)


class TestGetFold:
    def test_train_end_boundary_is_exclusive(self):
        """This is exactly the bug we fixed in train.py: the boundary row
        must land in val, not train.
        """
        df = make_valid_raw_df(n=72)
        train_end = pd.Timestamp("2012-01-02")
        val_start = pd.Timestamp("2012-01-02")
        val_end = pd.Timestamp("2012-01-02 23:00:00")

        train_fold, val_fold = get_fold(df, train_end, val_start, val_end)

        assert (train_fold["dteday"] < train_end).all()
        assert not (train_fold["dteday"] >= train_end).any()
        assert (val_fold["dteday"] >= val_start).all()
