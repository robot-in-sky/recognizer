from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import ENV_FILE_ENCODING, ENV_FILE_PATH, ENV_NESTED_DELIMITER
from src.io.rabbitmq import RabbitMQSettings
from src.io.s3 import S3Settings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding=ENV_FILE_ENCODING,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
    )

    rabbitmq: RabbitMQSettings
    s3: S3Settings

    debug: bool = False
    log_level: str = "INFO"
