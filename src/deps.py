from dataclasses import dataclass

from src.services.queue import QueueDispatcher
from src.services.recognition import DocumentRecognizer
from src.services.storage import AttachmentStorage
from src.settings import AppSettings


@dataclass
class Dependencies:
    settings: AppSettings
    dispatcher: QueueDispatcher
    storage: AttachmentStorage
    recognizer: DocumentRecognizer
