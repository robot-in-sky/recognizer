from src.io.rabbitmq import RabbitMQSettings
from src.io.s3 import S3Settings
from src.services.mq import MQSettings

from .base_app_settings import BaseAppSettings
from .log_settings import LogSettings


class AppSettings(BaseAppSettings):

    # Required settings
    rabbitmq: RabbitMQSettings
    s3: S3Settings

    # Optional settings
    mq: MQSettings = MQSettings()
    log: LogSettings = LogSettings()
