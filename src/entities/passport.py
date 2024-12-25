from datetime import date
from enum import StrEnum, auto

from src.entities.base import Base


class Gender(StrEnum):
    MALE = auto()
    FEMALE = auto()


class PassportDetails(Base):
    surname: str
    given_name: str
    passport_no: str
    country: str
    birth_date: date
    gender: Gender
    birth_place: str
    issue_date: date
    expire_date: date


class PassportAttachments(Base):
    scanned: str
    photo: str
    debug: str
