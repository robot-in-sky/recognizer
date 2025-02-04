from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.entities.base import Base


class CaptchaDetails(Base):
    captcha: str | None


@dataclass
class CaptchaImages:
    debug: np.ndarray | None


class CaptchaFiles(StrEnum):
    SOURCE = "source.png"
    DEBUG = "debug.png"
