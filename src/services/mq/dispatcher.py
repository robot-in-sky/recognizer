import logging
from collections.abc import Callable, Coroutine
from typing import Any

import aio_pika
from pydantic import ValidationError

from src.entities.recognition import RecognitionRequest, RecognitionResult
from src.io import RabbitMQClient

from .dispatcher_settings import DispatcherSettings


class Dispatcher:

    def __init__(self, client: RabbitMQClient, settings: DispatcherSettings) -> None:
        self.client = client
        self.settings = settings
        self._logger = logging.getLogger(__name__)


    async def consume( self,
            request_handler: Callable[[RecognitionRequest],
                                        Coroutine[Any, Any, RecognitionResult | None]],
    ) -> None:
        queue_name = self.settings.request_queue_name

        async def on_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with message.process():
                try:
                    request = RecognitionRequest.model_validate_json(message.body.decode())
                except ValidationError:
                    self._logger.warning("Invalid request body: %s", message.body.decode())
                else:
                    result = await request_handler(request)
                    if result is not None:
                        await self.publish(result)

        await self.client.consume(queue_name, on_message)


    async def publish(self, result: RecognitionResult) -> None:
        base = self.settings.result_queue_base
        result_type = str(result.type).lower()
        queue_name = f"{base}.{result_type}"
        message = aio_pika.Message(body=result.model_dump_json().encode())
        await self.client.publish(queue_name, message)
