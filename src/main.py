import asyncio
import logging
from datetime import UTC, datetime

import aio_pika
import cv2
from uuid_extensions import uuid7

from src.entities.passport import PassportAttachments
from src.entities.recognition import RecognitionRequest, RecognitionResult, RecognitionStatus, RecognitionType
from src.io import RabbitMQClient, S3Client
from src.services.mq import Dispatcher
from src.services.recognition import DocumentRecognizer
from src.services.storages import AttachmentStorage
from src.settings import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings()
rabbitmq_client = RabbitMQClient(settings.rabbitmq)
dispatcher = Dispatcher(rabbitmq_client, settings.dp)
s3_client = S3Client(settings.s3)
storage = AttachmentStorage(s3_client)
recognizer = DocumentRecognizer()


async def make_request() -> None:
    request_id = uuid7()
    image = cv2.imread("../tests/input/2.jpg")
    source = await storage.add_image(str(request_id), "source.png", image)
    message = aio_pika.Message(
        RecognitionRequest(
            id=request_id,
            timestamp=int(datetime.now(UTC).timestamp()),
            type=RecognitionType.PASSPORT,
            source=source,
        ).model_dump_json().encode())
    await rabbitmq_client.publish(settings.dp.request_queue_name, message)


async def handle_request(request: RecognitionRequest) -> RecognitionResult | None:
    if request.type != RecognitionType.PASSPORT:
        return None
    start_time = datetime.now(UTC)
    source = await storage.get_image(str(request.id), "source.png")
    details, images = recognizer.process_image(source)
    status, message = RecognitionStatus.SUCCESS, "ОК"
    scanned, photo, debug = await asyncio.gather(
        storage.add_image(str(request.id), "scanned.png", images.scanned),
        storage.add_image(str(request.id), "photo.png", images.photo),
        storage.add_image(str(request.id), "debug.png", images.debug),
    )
    duration = datetime.now(UTC) - start_time
    print(details)
    print(duration)

    attachments = PassportAttachments(source=request.source,
                                      scanned=scanned,
                                      photo=photo,
                                      debug=debug)

    return RecognitionResult(request_id=request.id,
                             request_timestamp=request.timestamp,
                             type=request.type,
                             status=status,
                             message=message,
                             duration=duration,
                             details=details,
                             attachments=attachments)


async def main() -> None:
    await make_request()
    await dispatcher.consume(handle_request)


if __name__ == "__main__":
    asyncio.run(main())
