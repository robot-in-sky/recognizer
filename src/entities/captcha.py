from dataclasses import dataclass

import numpy as np

from src.entities.base import Base


class CaptchaDetails(Base):
    captcha: str | None


@dataclass
class CaptchaImages:
    debug: np.ndarray | None


class CaptchaAttachments(Base):
    source: str
    debug: str | None
