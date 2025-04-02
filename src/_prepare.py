import asyncio
from datetime import UTC, datetime
from pathlib import Path

import cv2
from src.config import ROOT_DIR
from src.services.recognition import PassportRecognizer
from src.services.recognition.utils.image_grid import ImageGrid

SRC_PATH = Path(ROOT_DIR).joinpath("tests/unit/services/recognition/input/4.jpg")

async def main() -> None:
    recognizer = PassportRecognizer()

    img = cv2.imread(str(SRC_PATH))
    if img is None:
        msg = "Source image could not be read."
        raise ValueError(msg)

    start = datetime.now(UTC)

    grid_debug = ImageGrid(row_height=800)
    grid_debug.add(img, "Original")
    img = recognizer.scale_to_ref(img)
    img_corrected = recognizer.correct_skew_by_ref_features(img)

    corners_top, corners_bottom, img_detect_top, img_detect_bottom \
        = recognizer.detect_top_and_bottom_page_corners(img_corrected)
    grid_debug.add(img_corrected, "Corrected")
    grid_debug.add(img_detect_top, "Top page detection")
    grid_debug.add(img_detect_bottom, "Bottom page detection")

    img_debug = grid_debug.render()
    cv2.imshow("Debug", img_debug)

    end = datetime.now(UTC)

    td = (end - start).total_seconds() * 10 ** 3
    print(f"The time of execution is : {td:.03f}ms")
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(main())
