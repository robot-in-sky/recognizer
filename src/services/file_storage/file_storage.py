import random
import string
from datetime import UTC, datetime

from uuid_extensions import uuid7

from src.io.s3_client import S3Client


class FileStorage:

    def __init__(self, s3: S3Client, prefix: str = "") -> None:
        self.s3 = s3
        self.prefix = prefix

    @staticmethod
    def generate_group_id() -> str:
        dt = datetime.now(UTC).strftime("%Y%m%d%H%M")
        rnd = "".join(random.choices(string.ascii_letters +   # noqa: S311
                                     string.digits, k=6))
        return f"{dt}-{rnd}"

    def _get_object_name(self, name: str, group_id: str | None) -> str:
        if group_id is None:
            return name
        return f"{self.prefix}{group_id}-{name}"

    async def add(self, name: str, data: bytes, *, group_id: str | None = None) -> str:
        obj_name = self._get_object_name(name, group_id)
        await self.s3.put_object(obj_name, data)
        return obj_name

    async def get(self, name: str, *, group_id: str | None = None) -> bytes:
        obj_name = self._get_object_name(name, group_id)
        return await self.s3.get_object(obj_name)

    async def delete(self, name: str, *, group_id: str | None = None) -> None:
        obj_name = self._get_object_name(name, group_id)
        await self.s3.delete_object(obj_name)
