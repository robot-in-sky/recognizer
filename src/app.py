import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

from src.deps import Dependencies
from src.entities.passport import PassportAttachments
from src.entities.recognition import RecognitionRequest, RecognitionResponse, RecognitionStatus
from src.services.recognition import RecognitionError
from src.services.storage import StorageClientError


class Application:
    def __init__(self, deps: Dependencies) -> None:
        self.deps = deps
        self._logger = logging.getLogger(__name__)


    async def handle_request(self, request: RecognitionRequest) -> RecognitionResponse:
        request_id = str(request.id)
        start_time = datetime.now(UTC)

        try:
            source = await self.deps.storage.get_image(request_id, "source.png")

            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor() as executor:
                details, images = await loop.run_in_executor(
                    executor, self.deps.recognizer.process_image, source,
                )
            scanned, photo, debug = await asyncio.gather(
                self.deps.storage.add_image(request_id, "scanned.png", images.scanned),
                self.deps.storage.add_image(request_id, "photo.png", images.photo),
                self.deps.storage.add_image(request_id, "debug.png", images.debug),
            )

            duration = datetime.now(UTC) - start_time
            status, message = RecognitionStatus.SUCCESS, "ОК"
            self._logger.info(f"Recognition status: {status}, duration: {duration}")

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

        except RecognitionError as error:
            status, message = RecognitionStatus.ERROR, "Recognition error"
            self._logger.error(error)
            self._logger.error(f"Recognition status: {status}, message: {message}")

        except StorageClientError as error:
            status, message = RecognitionStatus.ERROR, "Storage client error"
            self._logger.error(error)
            self._logger.error(f"Recognition status: {status}, message: {message}")

        return RecognitionResponse(request_id=request.id,
                                   request_timestamp=request.timestamp,
                                   type=request.type,
                                   status=status,
                                   message=message)


    async def start(self) -> None:
        await self.deps.dispatcher.consume(self.handle_request)
