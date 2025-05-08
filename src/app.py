import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from uuid import UUID

from aio_pika.patterns import JsonRPC

from clients.amqp import AMQPClient
from clients.s3 import S3Client
from entities.passport import PassportFiles
from entities.recognition import RecognitionResult, RecognitionType
from services.recognition import PassportRecognizer
from services.storage import AttachmentStorage
from settings import AppSettings, LogSettings


class Application:
    def __init__(self, settings: AppSettings) -> None:
        self.configure_logging(settings.log)

        self.settings = settings
        self.storage = AttachmentStorage(S3Client(settings.s3))
        self.recognizer = PassportRecognizer()

        self._logger = logging.getLogger(__name__)


    @staticmethod
    def configure_logging(settings: LogSettings) -> None:
        logging.basicConfig(level=settings.level)

    """
    async def make_request(self) -> None:
        request_id = str(uuid7())
        image = cv2.imread(ROOT_DIR.joinpath("tests/input/2.jpg"))
        await self.storage.add_image(request_id, "source.png", image)
        result = await self.deps.rpc.proxy.recognize_passport(request_id=request_id)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(result)
    """


    async def recognize_passport(self, id_: str) -> dict[str, Any]:
        uuid_ = UUID(id_)
        source_img = await self.storage.get_image(uuid_, PassportFiles.SOURCE)

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            details, images = await loop.run_in_executor(
                executor, self.recognizer.process_image, source_img,
            )
        await asyncio.gather(
            self.storage.add_image(uuid_, PassportFiles.SCANNED, images.scanned),
            self.storage.add_image(uuid_, PassportFiles.PHOTO, images.photo),
            self.storage.add_image(uuid_, PassportFiles.DEBUG, images.debug),
        )

        return RecognitionResult(id=uuid_,
                                 type=RecognitionType.PASSPORT,
                                 details=details).model_dump(mode="json")


    async def start(self) -> None:
        amqp = AMQPClient(self.settings.amqp)
        connection = await amqp.get_connection()
        channel = await connection.channel()
        self._logger.info("Waiting for messages")
        rpc = await JsonRPC.create(channel)
        await rpc.register("recognize_passport", self.recognize_passport)
        try:
            await asyncio.Future()
        finally:
            await connection.close()
