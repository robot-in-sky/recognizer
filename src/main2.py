import asyncio
import json
import logging

from aio_pika import Message
from aio_pika.abc import AbstractIncomingMessage

from src.io import RabbitMQClient, S3Client
from src.settings import Settings

from uuid_extensions import uuid7

logging.basicConfig(level=logging.INFO)

settings = Settings()
broker = RabbitMQClient(settings.rabbitmq)
s3_client = S3Client(settings.s3)


async def handle_message(message: AbstractIncomingMessage) -> None:
    async with message.process():
        print("Type:", message.type)
        body = message.body.decode()
        print("Body:", body)
        try:
            data = json.loads(body)
            print("Data:", data)
        except json.JSONDecodeError:
            pass
        # await broker.publish("test_queue2", message)

async def main() -> None:
    await broker.publish("test_queue", Message(type="task", body=b"test111"))
    await broker.publish("test_queue", Message(type="task", body=b"test222"))
    await broker.publish("recognition.requests", Message(type="passport", body=json.dumps({"source": "sdfhyasdfyhasiogfhasdlghfdgbnm.png"}).encode()))
    await broker.publish("recognition.results", Message(type="passport", body=json.dumps({"source": "sdfhyasdfyhasiogfhasdlghfdgbnm.png"}).encode()))
    await broker.consume("test_queue", handle_message)

if __name__ == "__main__":
    asyncio.run(main())
