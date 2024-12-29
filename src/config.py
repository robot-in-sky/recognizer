from pathlib import Path

APP_NAME = "recognizer"

SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent

ENV_FILE_PATH = Path(ROOT_DIR).joinpath("env/.env")
ENV_FILE_ENCODING = "utf-8"
ENV_NESTED_DELIMITER = "__"
