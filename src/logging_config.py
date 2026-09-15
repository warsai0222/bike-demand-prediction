import logging

from src import config


def setup_logging(level=logging.INFO):
    """Central logging setup, called once from each module's entrypoint.
    Logs go to both the console and a shared log file (config.LOG_FILE_PATH),
    so you can tail one file to see everything the pipeline did across
    features -> train -> evaluate -> monitor, in the order it happened.
    """
    config.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE_PATH),
        ],
    )
