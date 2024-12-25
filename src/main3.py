import asyncio
from datetime import UTC, datetime

import cv2

from src.services.cv import DocumentRecognizer

SRC_PATH = "../tests/input/2.jpg"

async def main() -> None:
    recognizer = DocumentRecognizer()

    source = cv2.imread(SRC_PATH)

    start = datetime.now(UTC)
    values, scan, photo, output = recognizer.process_image(source)
    end = datetime.now(UTC)

    td = (end - start).total_seconds() * 10 ** 3
    print(f"The time of execution of above program is : {td:.03f}ms")

    for k, v in values.items():
        print(k, ":", v)

    cv2.imshow("Scan", scan)
    cv2.imshow("Photo", photo)
    cv2.imshow("Pipeline", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
