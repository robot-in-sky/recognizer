import asyncio
from datetime import UTC, datetime

import cv2
import numpy as np

from src.services.cv import DocumentRecognizer
from src.services.file_storage import FileStorage, S3Client
from src.settings import AppSettings

SRC_PATH = "../tests/input/2.jpg"

async def main() -> None:
    settings = AppSettings()
    # print(settings.s3)
    storage = FileStorage(S3Client(settings.s3))
    source = cv2.imread(SRC_PATH)

    start = datetime.now(UTC)
    encoded = cv2.imencode(".png", source, [])[1]
    group_id = storage.generate_group_id()
    await storage.add("original.png", bytes(encoded), group_id=group_id)
    end = datetime.now(UTC)
    td = (end - start).total_seconds() * 10 ** 3
    print(f"storage.add : {td:.03f}ms")

    start = datetime.now(UTC)
    recognizer = DocumentRecognizer()
    values, scan, photo, pipeline = recognizer.process_image(source)
    end = datetime.now(UTC)
    td = (end - start).total_seconds() * 10 ** 3
    print(f"recognizer.process_image : {td:.03f}ms")

    print(values)
    encoded = cv2.imencode(".png", scan)[1]
    await storage.add("scan.png", bytes(encoded), group_id=group_id)
    encoded = cv2.imencode(".png", photo)[1]
    await storage.add("photo.png", bytes(encoded), group_id=group_id)
    encoded = cv2.imencode(".png", pipeline)[1]
    await storage.add("pipeline.png", bytes(encoded), group_id=group_id)

    start = datetime.now(UTC)
    data = await storage.get("scan.png", group_id=group_id)
    buf = np.frombuffer(data, dtype=np.int8)
    img_output = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imshow("Scan", img_output)
    end = datetime.now(UTC)
    td = (end - start).total_seconds() * 10 ** 3
    print(f"storage.get : {td:.03f}ms")

    data = await storage.get("photo.png", group_id=group_id)
    buf = np.frombuffer(data, dtype=np.int8)
    img_output = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imshow("Photo", img_output)

    data = await storage.get("pipeline.png", group_id=group_id)
    buf = np.frombuffer(data, dtype=np.int8)
    img_output = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.imshow("Pipeline", img_output)
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
