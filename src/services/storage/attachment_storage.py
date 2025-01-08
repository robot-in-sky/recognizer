import cv2
import numpy as np

from src.io.s3 import S3Client


class AttachmentStorage:

    def __init__(self, client: S3Client) -> None:
        self._client = client

    async def add_image(self, group_id: str, filename: str, image: np.ndarray) -> str:
        encoded = self._encode_image(image)
        return await self.add_bytes(group_id, filename, encoded)

    async def get_image(self, group_id: str, filename: str) -> np.ndarray:
        data = await self.get_bytes(group_id, filename)
        return self._decode_image(data)

    async def add_bytes(self, group_id: str, filename: str, data: bytes) -> str:
        obj_name = self._get_object_name(group_id, filename)
        await self._client.put_object(obj_name, data)
        return obj_name

    async def get_bytes(self, group_id: str, filename: str) -> bytes:
        obj_name = self._get_object_name(group_id, filename)
        return await self._client.get_object(obj_name)

    async def delete(self, group_id: str, filename: str) -> None:
        obj_name = self._get_object_name(group_id, filename)
        await self._client.delete_object(obj_name)

    @staticmethod
    def _get_object_name(group_id: str, filename: str) -> str:
        return f"{group_id}/{filename}"

    @staticmethod
    def _encode_image(image: np.ndarray) -> bytes:
        return bytes(cv2.imencode(".png", image, [])[1])

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray:
        buf = np.frombuffer(data, dtype=np.int8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
