.PHONY: install featurize train evaluate monitor pipeline test mlflow-ui sync-model clean

install:
	pip install -r requirements.txt

# Individual pipeline stages -- useful when iterating on just one piece
featurize:
	python -m src.features

train:
	python -m src.train

evaluate:
	python -m src.evaluate

monitor:
	python -m src.monitor

# Full DVC-tracked pipeline: only reruns stages whose deps/params/code changed
pipeline:
	dvc repro

test:
	pytest tests/ -v

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Sync the git-committed model artifact after a retrain (see script docstring)
sync-model:
	python scripts/sync_deployed_model.py

clean:
	rm -rf reports/figures/*.png
	rm -f logs/project.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
