# Bike Demand Prediction

End-to-end ML project predicting hourly bike-share demand, using the UCI
Bike Sharing (hourly) dataset. Covers data validation, leakage-safe feature
engineering, walk-forward model validation, experiment tracking, a DVC data
pipeline, automated testing, and simulated drift monitoring with a retrain
trigger.

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
  predict.py              shared prediction logic (used by monitor.py and any future serving layer)
  monitor.py               simulated drift monitoring + retrain trigger
  logging_config.py        centralized logging setup (console + logs/project.log)
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

**Note:** this dataset is historically frozen (ends Dec 2012) -- there is no
live feed of new data. `monitor.py` demonstrates the actual monitoring and
retrain-trigger *mechanism* using simulated replay of held-out data; a
production deployment would swap `load_stream_data()` for a real source of
incoming predictions/actuals, with everything else unchanged.

## Out of scope

This project deliberately stops at a working, testable, monitored ML
pipeline running locally. It does not include: containerization (Docker),
a cloud-hosted DVC remote, a REST/API serving layer, or CI/CD deployment
automation. A lightweight CI workflow (`.github/workflows/ci.yml`) runs the
test suite on push/PR, but there is no deployment pipeline beyond that.
