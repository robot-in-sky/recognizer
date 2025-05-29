from dataclasses import dataclass
from datetime import date
from enum import StrEnum, auto

import numpy as np

from entities.base import Base


class PassportType(StrEnum):
    RUS_5Y = auto()
    RUS_10Y = auto()
    OTHER = auto()


class Gender(StrEnum):
    MALE = auto()
    FEMALE = auto()


class PassportDetails(Base):
    surname: str | None = None
    given_name: str | None = None
    passport_no: str | None = None
    country: str | None = None
    nationality: str | None = None
    birth_date: date | None = None
    gender: Gender | None = None
    birth_place: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None


@dataclass
class PassportImages:
    scanned: np.ndarray | None
    photo: np.ndarray | None
    debug: np.ndarray | None


class PassportFiles(StrEnum):
    SOURCE = "source.png"
    SCANNED = "scanned.png"
    PHOTO = "photo.png"
    DEBUG = "debug.png"
