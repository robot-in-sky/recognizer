import asyncio
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import aio_pika
import cv2
from uuid_extensions import uuid7

from src.config import ROOT_DIR
from src.entities.passport import PassportAttachments
from src.entities.recognition import RecognitionRequest, RecognitionResponse, RecognitionStatus, RecognitionType
from src.io import RabbitMQClient, S3Client
from src.services.mq import MQDispatcher
from src.services.recognition import DocumentRecognizer
from src.services.storages import AttachmentStorage
from src.settings import AppSettings, ValidationError

AppSettings.save_dotenv_example_if_not_exists()

try:
    settings = AppSettings()
except ValidationError as e:
    logging.error(e)
    sys.exit(1)

settings.log.configure_logging()

rabbitmq_client = RabbitMQClient(settings.rabbitmq)
mq_dispatcher = MQDispatcher(rabbitmq_client, settings.mq)

s3_client = S3Client(settings.s3)
storage = AttachmentStorage(s3_client)

recognizer = DocumentRecognizer()


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
    await rabbitmq_client.publish(settings.mq.request_queue_name, message)


async def handle_request(request: RecognitionRequest) -> RecognitionResponse | None:
    if request.type != RecognitionType.PASSPORT:
        return None
    request_id = str(request.id)
    start_time = datetime.now(UTC)
    source = await storage.get_image(request_id, "source.png")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        details, images = await loop.run_in_executor(
            executor, recognizer.process_image, source,
        )

    status, message = RecognitionStatus.SUCCESS, "ОК"
    scanned, photo, debug = await asyncio.gather(
        storage.add_image(request_id, "scanned.png", images.scanned),
        storage.add_image(request_id, "photo.png", images.photo),
        storage.add_image(request_id, "debug.png", images.debug),
    )
    duration = datetime.now(UTC) - start_time
    print(details)
    print(duration)

    attachments = PassportAttachments(source=request.source,
                                      scanned=scanned,
                                      photo=photo,
                                      debug=debug)

    return RecognitionResponse(request_id=request.id,
                               request_timestamp=request.timestamp,
                               type=request.type,
                               status=status,
                               message=message,
                               duration=duration,
                               details=details,
                               attachments=attachments)


async def main() -> None:
    await make_request()
    await mq_dispatcher.consume(handle_request)


if __name__ == "__main__":
    asyncio.run(main())
