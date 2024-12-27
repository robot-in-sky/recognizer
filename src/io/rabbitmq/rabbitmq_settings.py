from aio_pika.connection import make_url
from pydantic import BaseModel
from yarl import URL


class RabbitMQSettings(BaseModel):
    host: str = "localhost"
    port: int | None = 5672
    username: str = "guest"
    password: str | None = "guest"
    ssl: bool = False

    reconnect_interval: int = 5
    durable_queues: bool = False

    @property
    def url(self) -> URL:
        return make_url(
            host=self.host,
            port=self.port,
            login=self.username,
            password=self.password,
            ssl=self.ssl)
