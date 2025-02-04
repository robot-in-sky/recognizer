from pydantic import BaseModel


class LogSettings(BaseModel):
    level: str = "INFO"
