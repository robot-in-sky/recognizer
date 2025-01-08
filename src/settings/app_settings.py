from src.io.amqp import AMQPSettings
from src.io.s3 import S3Settings
from src.settings.app_settings_base import AppSettingsBase
from src.settings.log_settings import LogSettings


class AppSettings(AppSettingsBase):

    # Required settings
    amqp: AMQPSettings
    s3: S3Settings

    # Optional settings
    log: LogSettings = LogSettings()

    request_queue_name: str | None = "recognition.request"
    response_routing_base: str | None = "recognition.response"


if __name__ == "__main__":
    AppSettings.save_dotenv_example()
