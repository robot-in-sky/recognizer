import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import aio_pika

from src.settings import RabbitMQSettings


class RabbitMQClient:

    def __init__(self, settings: RabbitMQSettings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)


    async def get_connection(self) -> aio_pika.abc.AbstractRobustConnection:
        connection = await aio_pika.connect_robust(
            url=self.settings.url,
            reconnect_interval = self.settings.reconnect_interval)
        self.logger.info("Connected to RabbitMQ")
        return connection


    async def wait_for_connection(self) -> aio_pika.abc.AbstractRobustConnection:
        try:
            return await self.get_connection()
        except aio_pika.exceptions.CONNECTION_EXCEPTIONS as e:
            self.logger.error(e)
            await asyncio.sleep(self.settings.reconnect_interval)
            return await self.wait_for_connection()


    async def consume(
            self,
            queue_name: str,
            on_message: Callable[[aio_pika.abc.AbstractIncomingMessage],
                            Coroutine[Any, Any, None]], *,
            prefetch_count: int = 1,
            no_ack: bool = False,
    ) -> None:
        connection = await self.wait_for_connection()
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=prefetch_count)
            queue = await channel.declare_queue(
                queue_name, durable=self.settings.durable_queues)
            await queue.consume(on_message, no_ack=no_ack)
            self.logger.info("Waiting for messages")
            await asyncio.Future()


    async def publish(
            self,
            queue_name: str,
            message: aio_pika.abc.AbstractMessage, *,
            timeout: float | None = None,
    ) -> None:
        async with asyncio.timeout(timeout):
            connection = await self.wait_for_connection()
            async with connection:
                channel = await connection.channel()
                queue = await channel.declare_queue(
                    queue_name, durable=self.settings.durable_queues)
                await channel.default_exchange.publish(
                    message, routing_key=queue.name)
                self.logger.info("Message sent")
