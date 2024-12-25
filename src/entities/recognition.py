from enum import StrEnum, auto
from uuid import UUID

from src.entities.base import Base
from src.entities.passport import PassportAttachments, PassportDetails


class RecognitionType(StrEnum):
    PASSPORT = auto()
    CAPTCHA = auto()


class RecognitionRequest(Base):
    id: UUID
    type: RecognitionType
    source: str
    timestamp: int


class RecognitionStatus(StrEnum):
    SUCCESS = auto()
    ERROR = auto()


class RecognitionResult(Base):
    request_id: UUID
    type: RecognitionType
    source: str
    request_timestamp: int
    status: RecognitionStatus
    message: str
    data: PassportDetails | None = None
    attachments: PassportAttachments | None = None
    duration: int
