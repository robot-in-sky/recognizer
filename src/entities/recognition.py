from enum import StrEnum, auto
from uuid import UUID

from entities.base import Base
from entities.captcha import CaptchaDetails
from entities.passport import PassportDetails


class RecognitionType(StrEnum):
    PASSPORT = auto()
    CAPTCHA = auto()


RecognitionDetails = PassportDetails | CaptchaDetails


class RecognitionResult(Base):
    id: UUID
    type: RecognitionType
    details: RecognitionDetails
