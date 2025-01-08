from datetime import timedelta
from enum import StrEnum, auto
from uuid import UUID

from src.entities.base import Base
from src.entities.captcha import CaptchaAttachments, CaptchaDetails
from src.entities.passport import PassportAttachments, PassportDetails


class RecognitionType(StrEnum):
    PASSPORT = auto()
    CAPTCHA = auto()


class RecognitionRequest(Base):
    id: UUID
    timestamp: int
    type: RecognitionType
    source: str


RecognitionDetails = PassportDetails | CaptchaDetails
RecognitionAttachments = PassportAttachments | CaptchaAttachments


class RecognitionStatus(StrEnum):
    SUCCESS = auto()
    ERROR = auto()


class RecognitionResponse(Base):
    request_id: UUID
    request_timestamp: int
    type: RecognitionType
    status: RecognitionStatus
    message: str
    duration: timedelta | None
    details: RecognitionDetails | None = None
    attachments: RecognitionAttachments | None = None
