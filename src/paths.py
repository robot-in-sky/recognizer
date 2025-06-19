from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent
ENV_FILE_PATH = Path(ROOT_DIR).joinpath("env/.env")
