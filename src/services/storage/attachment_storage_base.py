from uuid import UUID

from src.io.s3 import S3Client


class AttachmentStorageBase:

    def __init__(self, client: S3Client) -> None:
        self._client = client

    @staticmethod
    def _get_object_name(id_: UUID, basename: str) -> str:
        return f"{id_}/{basename}"

    async def add_bytes(self, id_: UUID, basename: str, data: bytes) -> str:
        obj_name = self._get_object_name(id_, basename)
        await self._client.put_object(obj_name, data)
        return obj_name

    async def get_bytes(self, id_: UUID, basename: str) -> bytes:
        obj_name = self._get_object_name(id_, basename)
        return await self._client.get_object(obj_name)

    async def delete(self, id_: UUID, basename: str) -> None:
        obj_name = self._get_object_name(id_, basename)
        await self._client.delete_object(obj_name)
