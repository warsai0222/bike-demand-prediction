# Bike Demand Prediction

End-to-end ML project predicting hourly bike-share demand, using the UCI
Bike Sharing (hourly) dataset. Covers data validation, leakage-safe feature
engineering, walk-forward model validation, experiment tracking, a DVC data
pipeline, automated testing, simulated drift monitoring with a retrain
trigger, and an interactive Streamlit dashboard for prediction and
monitoring.

## Project structure

```
params.yaml            single source of truth: paths, split dates, feature list, hyperparams
dvc.yaml / dvc.lock     DVC pipeline definition (featurize -> train -> evaluate)
src/
  config.py             loads params.yaml, exposes typed constants
  data.py                raw data loading, validation, chronological split
  features.py            leakage-safe feature engineering
  train.py               walk-forward CV, final model fit, MLflow tracking, versioned save
  evaluate.py             test-set evaluation, naive baseline comparison, residual analysis
  predict.py              shared prediction logic (used by monitor.py, the dashboard, and any future serving layer)
  monitor.py               simulated drift monitoring + retrain trigger
  logging_config.py        centralized logging setup (console + logs/project.log)
app/
  streamlit_app.py         interactive dashboard: predict a new hour, monitor model performance
.streamlit/
  config.toml               dashboard theme
tests/
  test_features.py         feature engineering regression tests
  test_validation.py       raw-data validation + split-boundary tests
notebooks/
  test.ipynb                exploratory analysis + the original modeling walkthrough
reports/
  metrics.json              latest test-set metrics (DVC-tracked)
  figures/                  residual diagnostic plots
models/
  current.json              pointer to the currently deployed model file
  *.pkl                      versioned, timestamped model files (never overwritten)
monitoring/
  prediction_log.csv        append-only log of predictions/actuals/errors from the monitoring replay
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

Each stage can be run individually:

```bash
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.monitor
```

Or via DVC, which only reruns stages whose code, params, or data actually
changed since the last run:

```bash
dvc repro
```

A `Makefile` wraps all of the above:

```bash
make featurize
make train
make evaluate
make monitor
make pipeline     # dvc repro
make test         # pytest
make mlflow-ui    # opens the MLflow UI against mlflow.db
make clean        # removes generated logs/figures/__pycache__
```

## Data versioning (DVC)

Raw data and the processed features are tracked with DVC. `dvc.yaml` defines
three pipeline stages: `featurize` -> `train` -> `evaluate`, each declaring
its own deps, tracked params (from `params.yaml`), and outputs. Metrics from
the `evaluate` stage are written to `reports/metrics.json` as a DVC `metrics`
output, so `dvc metrics show` / `dvc metrics diff` work across commits.

```bash
dvc repro       # reproduce the pipeline
dvc push        # push data/models to the configured remote
dvc metrics show
```

## Experiment tracking (MLflow)

Every `train.py` run logs params, per-fold and summary validation metrics,
and the trained model to MLflow (SQLite backend, `mlflow.db`). View runs with:

```bash
make mlflow-ui
```

## Model versioning

`train.py` never overwrites a model -- each run saves a new timestamped
`.pkl` under `models/`, and updates `models/current.json` to point at it.
Every other module (`evaluate.py`, `predict.py`, `monitor.py`) reads the
current model exclusively through that pointer, never a hardcoded filename.

## Testing

```bash
pytest tests/ -v
```

Tests focus on logic that's easy to silently break: leakage-safe lag/rolling
features, the rush-hour flag, raw-data validation rules, and chronological
split/fold boundary correctness (several tests exist specifically because
they're regression tests for real bugs found during development).

## Monitoring and drift

`src/monitor.py` simulates a production monitoring loop: it replays the
frozen test window in batches (as if new data were arriving), tracks a
rolling MAE, and compares it against a threshold derived from the model's
own validation MAE (`config.DRIFT_MAE_MULTIPLIER x baseline`). Exceeding
the threshold triggers a retrain via `dvc repro`.

Two flat files stand in for what a production system would keep in a
database: `models/current.json` is the "which model is live" registry
(filename, training timestamp, validation metrics -- read by every other
module instead of a hardcoded path), and `monitoring/prediction_log.csv`
is the append-only predictions table (`timestamp, actual, predicted,
abs_error`) that `monitor.py` writes to as it replays the test window and
that the dashboard reads back to compute rolling MAE for whatever date is
selected.

**Note:** this dataset is historically frozen (ends Dec 2012) -- there is no
live feed of new data. `monitor.py` demonstrates the actual monitoring and
retrain-trigger *mechanism* using simulated replay of held-out data; a
production deployment would swap `load_stream_data()` for a real source of
incoming predictions/actuals, with everything else unchanged.

## Dashboard

```bash
streamlit run app/streamlit_app.py
```

A single date (and hour) control drives the whole page. The top section,
**Predict a new hour**, builds a raw feature row from the date/hour/weather
you enter and calls `src.predict.predict_new_raw()` for a live estimate --
calendar fields (weekday, season, working day) are derived automatically
rather than asked for. Backtesting is supported for any date in the
historical record; predicting past the end of the recorded data is limited
to the single day right after it ends, since the model's lag features
(ride counts from 24h/168h prior) need real history to look back on --
forecasting further out would require a recursive multi-step loop that
compounds error and has no real future weather to condition on, so it's
deliberately not built (see "Deviations from the original plan" below).

Below that, **Monitoring, as of [the same date]** replays what
`src/monitor.py`'s simulation would have shown up through that point --
test-set MAE/RMSE/MAPE vs. the naive baseline, drift status against the
validation-derived threshold, residuals by hour, and residuals by weather/
holiday/working-day segment. Picking an earlier date shows the dashboard
as it would have looked mid-rollout; the monitoring section clamps to the
test window's start (Nov 1, 2012) if a backtest date is earlier than that.

## Deviations from the original plan

The original design called for a few things that ended up built
differently in practice. Noted honestly rather than left implicit:

- **Postgres -> CSV + JSON.** The plan specified a `predictions` table and
  optional `drift_flags` table in Postgres. What's built instead is a flat
  CSV log (`monitoring/prediction_log.csv`) and a JSON model registry
  (`models/current.json`) -- functionally the same monitoring story
  (append-only prediction history, rolling error, threshold comparison)
  without standing up and managing a database for a solo project. See
  "Monitoring and drift" above for how they're used.
- **Dashboard date range is wider than "replay-only."** The plan specified
  a dropdown populated only from dates a batch replay job had already
  processed, with the dashboard reading pre-computed predictions and never
  predicting on demand. The dashboard instead predicts on demand
  (`predict_new_raw()`) for any date in the historical record, and only
  falls back to a fixed, narrower window (the actual monitored test
  period) for the monitoring metrics themselves, which do require
  pre-computed replay data.
- **No live weather / future forecasting.** In line with the original
  plan's own explicit note that no live weather API or future-date
  forecasting should be used, the dashboard's date range for prediction is
  capped at one day past the historical record -- exactly as far as the
  model's lag features can reach using only real, recorded data.
- **MLflow tracks runs but isn't pushed through a formal registry stage.**
  Params, metrics, and the model itself are logged to every `train.py` run
  via MLflow (SQLite backend), viewable with `make mlflow-ui`, but there's
  no separate "register this run as the production model" step -- that
  role is filled by `models/current.json` instead.

## Future improvements

Honest list of what a production version of this would still need:

- Swap the CSV/JSON monitoring store for a real database (Postgres) so
  multiple processes can write predictions concurrently and query history
  efficiently at scale.
- A recursive multi-step forecasting mode (predict hour N+1, feed it back
  in as history, predict N+2, ...) to extend predictions meaningfully past
  the single-day boundary, with clear communication that accuracy degrades
  with horizon length and that weather for future hours is necessarily an
  assumption, not a forecast.
- Wire `monitor.py`'s retrain trigger to an actual scheduler instead of a
  manual/simulated run, and push newly trained models through a formal
  MLflow model registry stage (stage transitions, approval gating) rather
  than the current JSON pointer.
- A REST/API serving layer, containerization, and a cloud-hosted DVC
  remote -- all explicitly out of scope for this project (see below) but
  the natural next steps for turning this into a deployed system.

## Out of scope

This project deliberately stops at a working, testable, monitored ML
pipeline running locally. It does not include: containerization (Docker),
a cloud-hosted DVC remote, a REST/API serving layer, or CI/CD deployment
automation. A lightweight CI workflow (`.github/workflows/ci.yml`) runs the
test suite on push/PR, but there is no deployment pipeline beyond that.
