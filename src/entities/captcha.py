from src.entities.base import Base


class CaptchaDetails(Base):
    captcha: str | None


class CaptchaAttachments(Base):
    source: str
    debug: str | None
