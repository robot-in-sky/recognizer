from aio_pika.connection import make_url
from pydantic import BaseModel
from yarl import URL


class AMQPSettings(BaseModel):
    host: str
    port: int | None = 5672
    username: str = "guest"
    password: str | None = "guest"
    ssl: bool = False

    reconnect_interval: int = 5
    durable: bool = False

    @property
    def url(self) -> URL:
        return make_url(
            host=self.host,
            port=self.port,
            login=self.username,
            password=self.password,
            ssl=self.ssl)
