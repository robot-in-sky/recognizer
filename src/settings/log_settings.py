import logging

from pydantic import BaseModel


class LogSettings(BaseModel):
    level: str = "INFO"

    def configure_logging(self) -> None:
        logging.basicConfig(level=self.level)
