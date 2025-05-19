import asyncio
from datetime import UTC, datetime
from pathlib import Path

import cv2

from src.paths import ROOT_DIR
from src.services.recognition import PassportRecognizer

SRC_PATH = Path(ROOT_DIR).joinpath("tests/unit/services/recognition/input/1.jpg")

async def main() -> None:
    recognizer = PassportRecognizer()

    img = cv2.imread(str(SRC_PATH))
    if img is None:
        msg = "Source image could not be read."
        raise ValueError(msg)

    start = datetime.now(UTC)

    details, images = recognizer.process_image(img)
    end = datetime.now(UTC)

    td = (end - start).total_seconds() * 10 ** 3
    print(f"The time of execution is : {td:.03f}ms")
    print(details)

    cv2.imshow("Scan", images.scanned)
    cv2.imshow("Photo", images.photo)
    cv2.imshow("Debug", images.debug)
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
