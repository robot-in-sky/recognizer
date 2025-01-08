import asyncio
import logging
import sys
from datetime import UTC, datetime

import aio_pika
import cv2
from uuid_extensions import uuid7

from src.app import Application
from src.config import ROOT_DIR
from src.deps import Dependencies
from src.entities.recognition import RecognitionRequest, RecognitionType
from src.io.amqp import AMQPClient
from src.io.s3 import S3Client
from src.services.queue import QueueDispatcher
from src.services.recognition import DocumentRecognizer
from src.services.storage import AttachmentStorage
from src.settings import AppSettings, ValidationError


async def main() -> None:
    try:
        settings = AppSettings()
    except ValidationError as e:
        logging.error(e)
        sys.exit(1)

    settings.log.configure_logging()

    amqp_client = AMQPClient(settings.amqp)
    dispatcher = QueueDispatcher(amqp_client,
                                 settings.request_queue_name,
                                 settings.response_routing_base)

    s3_client = S3Client(settings.s3)
    storage = AttachmentStorage(s3_client)
    recognizer = DocumentRecognizer()

    deps = Dependencies(settings=settings,
                        dispatcher=dispatcher,
                        storage=storage,
                        recognizer=recognizer)

    app = Application(deps)


    async def make_request() -> None:
        request_id = uuid7()
        image = cv2.imread(ROOT_DIR.joinpath("tests/input/2.jpg"))
        source = await storage.add_image(str(request_id), "source.png", image)
        message = aio_pika.Message(
            RecognitionRequest(
                id=request_id,
                timestamp=int(datetime.now(UTC).timestamp()),
                type=RecognitionType.PASSPORT,
                source=source,
            ).model_dump_json().encode())
        await amqp_client.publish(settings.request_queue_name, message)

    await make_request()
    await app.start()



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Interrupted")
    except Exception:
        logging.exception("Unexpected error")
        sys.exit(1)
