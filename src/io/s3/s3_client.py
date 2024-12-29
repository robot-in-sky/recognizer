import logging
from contextlib import asynccontextmanager

from aiobotocore.session import ClientCreatorContext, get_session
from botocore.exceptions import ClientError

from .exceptions import S3ClientError
from .s3_settings import S3Settings


class S3Client:

    def __init__(self, settings: S3Settings) -> None:
        self.service_name = "s3"
        self.config = {
            "aws_access_key_id": settings.access_key,
            "aws_secret_access_key": settings.secret_key,
            "endpoint_url": settings.url,
        }
        self.bucket_name = settings.bucket_name
        self.session = get_session()
        self._logger = logging.getLogger(__name__)


    @asynccontextmanager
    async def get_client(self) -> ClientCreatorContext:
        async with self.session.create_client(
                self.service_name, **self.config) as client:
            yield client


    async def get_object(self, object_name: str) -> bytes:
        try:
            async with self.get_client() as client:
                response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
                data = await response["Body"].read()
                self._logger.info(f"Data {object_name} received")
                return data
        except ClientError as error:
            raise S3ClientError from error


    async def put_object(self, object_name: str, data: bytes) -> None:
        try:
            async with self.get_client() as client:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=data,
                )
                self._logger.info(f"File {object_name} uploaded to {self.bucket_name}")
        except ClientError as error:
            raise S3ClientError from error


    async def delete_object(self, object_name: str) -> None:
        try:
            async with self.get_client() as client:
                await client.delete_object(Bucket=self.bucket_name, Key=object_name)
                self._logger.info(f"File {object_name} deleted from {self.bucket_name}")
        except ClientError as error:
            raise S3ClientError from error
