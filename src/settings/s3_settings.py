from pydantic import BaseModel


class S3Settings(BaseModel):
    url: str
    bucket_name: str
    access_key: str
    secret_key: str
