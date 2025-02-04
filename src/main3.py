import asyncio
from datetime import UTC, datetime

import cv2

from src.services.recognition import PassportRecognizer

SRC_PATH = "../tests/input/2.jpg"

async def main() -> None:
    recognizer = PassportRecognizer()

    source = cv2.imread(SRC_PATH)

    start = datetime.now(UTC)
    details, images = recognizer.process_image(source)
    end = datetime.now(UTC)

    td = (end - start).total_seconds() * 10 ** 3
    print(f"The time of execution is : {td:.03f}ms")

    # for k, v in details.items():
    #     print(k, ":", v)
    print(details)

    cv2.imshow("Scan", images.scanned)
    cv2.imshow("Photo", images.photo)
    cv2.imshow("Debug", images.debug)
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
