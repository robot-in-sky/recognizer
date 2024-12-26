from collections.abc import Callable, Coroutine
from typing import Any

import aio_pika

from src.entities.recognition import RecognitionRequest, RecognitionResult
from src.io import RabbitMQClient


class RecognitionQueueManager:

    def __init__(
            self,
            client: RabbitMQClient,
            callback: Callable[[RecognitionRequest],
                                    Coroutine[Any, Any, RecognitionResult]]) -> None:

        self._client = client
        self._callback = callback

        self.allowed_message_type = "recognition.request"
        self.request_queue = "recognition.requests"
        self.result_queue_base = "recognition.results"


    async def consume(self) -> None:
        await self._client.consume(self.request_queue, self._on_message)


    async def publish(self, result: RecognitionResult) -> None:
        result_type = str(result.type).lower()
        result_queue = f"{self.result_queue_base}.{result_type}"
        message = aio_pika.Message(body=result.model_dump_json().encode())
        await self._client.publish(result_queue, message)


    async def _on_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process():
            if message.type == self.allowed_message_type:
                request = RecognitionRequest.model_validate_json(message.body.decode())
                result = await self._callback(request)
                await self.publish(result)
