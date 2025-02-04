from enum import StrEnum, auto
from uuid import UUID

from src.entities.base import Base
from src.entities.captcha import CaptchaDetails
from src.entities.passport import PassportDetails


class RecognitionType(StrEnum):
    PASSPORT = auto()
    CAPTCHA = auto()


RecognitionDetails = PassportDetails | CaptchaDetails


class RecognitionResult(Base):
    id: UUID
    type: RecognitionType
    details: RecognitionDetails
