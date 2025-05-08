from dataclasses import dataclass
from datetime import date
from enum import StrEnum, auto

import numpy as np

from entities.base import Base


class Gender(StrEnum):
    MALE = auto()
    FEMALE = auto()


class PassportDetails(Base):
    surname: str | None
    given_name: str | None
    passport_no: str | None
    country: str | None
    birth_date: date | None
    gender: Gender | None
    birth_place: str | None
    issue_date: date | None
    expiry_date: date | None


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
