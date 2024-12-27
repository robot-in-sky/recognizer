from src.io.s3.s3_client import S3Client


class AttachmentStorage:

    def __init__(self, client: S3Client) -> None:
        self._client = client

    @staticmethod
    def _get_object_name(recognition_id: str, filename: str) -> str:
        return f"{recognition_id}/{filename}"

    async def add(self, recognition_id: str, filename: str, data: bytes) -> str:
        obj_name = self._get_object_name(recognition_id, filename)
        await self._client.put_object(obj_name, data)
        return obj_name

    async def get(self, recognition_id: str, filename: str) -> bytes:
        obj_name = self._get_object_name(recognition_id, filename)
        return await self._client.get_object(obj_name)

    async def delete(self, recognition_id: str, filename: str) -> None:
        obj_name = self._get_object_name(recognition_id, filename)
        await self._client.delete_object(obj_name)
