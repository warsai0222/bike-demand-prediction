.PHONY: install featurize train evaluate monitor pipeline test mlflow-ui clean

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

clean:
	rm -rf reports/figures/*.png
	rm -f logs/project.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
