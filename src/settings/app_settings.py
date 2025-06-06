from .amqp_settings import AMQPSettings
from .base.app_settings_base import AppSettingsBase
from .log_settings import LogSettings
from .s3_settings import S3Settings


class AppSettings(AppSettingsBase):

    # Required settings
    amqp: AMQPSettings
    s3: S3Settings

    # Optional settings
    run_in_executor: bool = True
    log: LogSettings = LogSettings()


if __name__ == "__main__":
    AppSettings.save_dotenv_example()
