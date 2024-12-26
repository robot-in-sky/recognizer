from src.io.s3_client import S3Client


class AttachmentStorage:

    def __init__(self, s3: S3Client, prefix: str = "") -> None:
        self.s3 = s3
        self.prefix = prefix

    @staticmethod
    def _get_object_name(recognition_id: str, filename: str) -> str:
        return f"{recognition_id}/{filename}"

    async def add(self, recognition_id: str, filename: str, data: bytes) -> str:
        obj_name = self._get_object_name(recognition_id, filename)
        await self.s3.put_object(obj_name, data)
        return obj_name

    async def get(self, recognition_id: str, filename: str) -> bytes:
        obj_name = self._get_object_name(recognition_id, filename)
        return await self.s3.get_object(obj_name)

    async def delete(self, recognition_id: str, filename: str) -> None:
        obj_name = self._get_object_name(recognition_id, filename)
        await self.s3.delete_object(obj_name)
