from src.infrastructure.amqp import AMQPSettings
from src.infrastructure.s3 import S3Settings
from src.settings.app_settings_base import AppSettingsBase
from src.settings.log_settings import LogSettings


class AppSettings(AppSettingsBase):

    # Required settings
    amqp: AMQPSettings
    s3: S3Settings

    # Optional settings
    log: LogSettings = LogSettings()


if __name__ == "__main__":
    AppSettings.save_dotenv_example()
