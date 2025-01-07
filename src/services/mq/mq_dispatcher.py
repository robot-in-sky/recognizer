import logging
from collections.abc import Callable, Coroutine
from typing import Any

import aio_pika
from pydantic import ValidationError

from src.entities.recognition import RecognitionRequest, RecognitionResponse
from src.io import RabbitMQClient

from .mq_settings import MQSettings


class MQDispatcher:

    def __init__(self, client: RabbitMQClient, settings: MQSettings) -> None:
        self.client = client
        self.settings = settings
        self._logger = logging.getLogger(__name__)


    async def consume( self,
            request_handler: Callable[[RecognitionRequest],
                                        Coroutine[Any, Any, RecognitionResponse | None]],
    ) -> None:
        queue_name = self.settings.request_queue_name

        async def on_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with message.process():
                try:
                    request = RecognitionRequest.model_validate_json(message.body.decode())
                except ValidationError:
                    self._logger.warning(
                        "Invalid request body: %s", message.body.decode())
                else:
                    result = await request_handler(request)
                    if result is not None:
                        await self.publish(result)

        await self.client.consume(queue_name, on_message)


    async def publish(self, response: RecognitionResponse) -> None:
        routing_base = self.settings.response_routing_base
        response_type = str(response.type).lower()
        routing_key = f"{routing_base}.{response_type}"
        message = aio_pika.Message(body=response.model_dump_json().encode())
        await self.client.publish(routing_key, message)
