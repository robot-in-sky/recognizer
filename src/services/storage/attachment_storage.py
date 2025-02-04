from uuid import UUID

import cv2
import numpy as np

from .attachment_storage_base import AttachmentStorageBase


class AttachmentStorage(AttachmentStorageBase):

    @staticmethod
    def _encode_image(image: np.ndarray) -> bytes:
        return bytes(cv2.imencode(".png", image, [])[1])

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray:
        buf = np.frombuffer(data, dtype=np.int8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    async def add_image(self, id_: UUID, basename: str, image: np.ndarray) -> str:
        encoded = self._encode_image(image)
        return await self.add_bytes(id_, basename, encoded)

    async def get_image(self, id_: UUID, basename: str) -> np.ndarray:
        data = await self.get_bytes(id_, basename)
        return self._decode_image(data)
