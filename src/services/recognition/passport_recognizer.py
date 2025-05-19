import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from numpy import ndarray
from svgelements import SVG

from entities.passport import Gender, PassportDetails, PassportImages
from src.services.keypoints.keypoints_detector import KeypointsDetector
from src.services.keypoints.validate import validate_pages

from .exceptions import CornerDetectionError, TextRecognitionError
from .utils.filters import clahe_filter, sharp_filter
from .utils.image_grid import ImageGrid
from .utils.ocr_helpers import detect_box
from .utils.shapes import Point, Rectangle
from .utils.skew_correction import adjust_corners


class PassportRecognizer:
    def __init__(self) -> None:

        ref_dir = Path(__file__).parent.joinpath("ref/").resolve()
        ref_mask_path = Path(ref_dir).joinpath("passport.svg")

        self._ref_mask = SVG().parse(str(ref_mask_path))

        frame_el = self._ref_mask.get_element_by_id("frame")
        self._frame = Rectangle(frame_el.x, frame_el.y, frame_el.width, frame_el.height)
        self._frame_top = Rectangle(self._frame.x, self._frame.y,
                                    self._frame.width, self._frame.height / 2)
        self._frame_bottom = Rectangle(self._frame.x, self._frame.y + self._frame.height / 2,
                                        self._frame.width, self._frame.height / 2)

        self._keypoints_detector = KeypointsDetector()
        self.scan_margin = (40, 40)

        self.roi_ids = [
            "surname",
            "given_name",
            "passport_no",
            "country",
            "birth_date",
            "gender",
            "birth_place",
            "issue_date",
            "expiry_date",
            "photo",
        ]
        self.roi_upscale = 2

        self.tesseract_config_eng = \
            "--psm 7 -c tessedit_char_whitelist=' ./-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
        self.tesseract_config_rus = \
            "--psm 7 -c tessedit_char_whitelist=' ./-1234567890АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'"


    def detect_corners(self, img: np.ndarray) -> tuple[list[Point], np.ndarray]:
        detected = self._keypoints_detector.predict_keypoints(img)
        orig_height, orig_width = img.shape[:2]
        if not validate_pages(detected, (orig_width, orig_height)):
            raise CornerDetectionError
        img_detect = img.copy()
        corners = [Point(d[0], d[1]) for d in detected]
        self.draw_markers(img_detect, corners)
        return corners, img_detect


    def adjust_pages_by_ref_frames(
            self, img: np.ndarray,
            corners_top: Sequence[Point],
            corners_bottom: Sequence[Point]) -> tuple[ndarray, list[Point], list[Point]]:
        # Adjust by frames and crop images with margin
        page_top, corners_top = adjust_corners(
            img, corners_top, self._frame_top,
            margin=(self.scan_margin[1], self.scan_margin[0], 0, self.scan_margin[0]))

        page_bottom, corners_bottom = adjust_corners(
            img, corners_bottom, self._frame_bottom,
            margin=(0, self.scan_margin[0], self.scan_margin[1], self.scan_margin[0]))

        for i in range(len(corners_bottom)):
            corners_bottom[i].y += page_top.shape[0]

        # Stack top and bottom pages
        return np.vstack((page_top, page_bottom)), corners_top, corners_bottom


    @staticmethod
    def draw_markers(
            img: np.ndarray,
            markers: Sequence[Point]) -> None:
        for p in markers:
            cv2.circle(img, p.tup, 8, (0, 255, 255), -1)


    @staticmethod
    def apply_adoptive_contrast(img: np.ndarray) -> np.ndarray:
        return clahe_filter(img, clip_limit=0.5)


    @staticmethod
    def apply_sharp_filter(img: np.ndarray) -> np.ndarray:
        return sharp_filter(img)


    def extract_rois(self, img: np.ndarray, *, draw: bool = False) -> dict[str, np.ndarray]:
        # Extract regions from source image
        ims_roi = {}
        for id_ in self.roi_ids:
            if el := self._ref_mask.get_element_by_id(id_):
                # Get rectangle from SVG mask
                rect = Rectangle(el.x - self._frame.x + self.scan_margin[0],
                                 el.y - self._frame.y + self.scan_margin[1],
                                 el.width, el.height)
                # Extract region
                (x1, y1), (x2, y2) = rect.p1.tup, rect.p3.tup
                ims_roi[id_] = np.copy(img[y1:y2, x1:x2])
                if draw:
                    cv2.rectangle(img, rect.p1.tup, rect.p3.tup, (255, 0, 0), 2)
        return ims_roi


    def extract_text_boxes(self, rois: dict[str, np.ndarray],
                           *, draw: bool = False) -> dict[str, np.ndarray]:
        # Extract text boxes from regions
        boxes = {}
        for id_ in self.roi_ids:
            if id_ == "photo":
                continue
            img = rois[id_]
            # Get bounding box
            min_h = img.shape[0] * 0.4
            box = detect_box(img, min_h)
            # Crop by box and apply upscale
            (x1, y1), (x2, y2) = box.p1.tup, box.p3.tup
            box_img = np.copy(img[y1:y2, x1:x2])
            boxes[id_] = cv2.resize(box_img, None,
                                    fx=self.roi_upscale, fy=self.roi_upscale,
                                    interpolation=cv2.INTER_CUBIC)
            if draw:
                cv2.rectangle(img, box.p1.tup, box.p3.tup, (255, 0, 0), 1)
        return boxes


    def recognize_field(self, id_: str, img: np.ndarray) -> Any:  # noqa: PLR0911
        text = pytesseract.image_to_string(
            img, lang="eng", config=self.tesseract_config_eng)
        text = text.replace("\n\x0c", "")
        text = text.replace("\n", "")

        match id_:
            case id_ if id_ in ["surname", "given_name"]:
                return re.sub(r"[^A-Z]", "", text)

            case "passport_no":
                return re.sub(r"[^0-9\s]", "", text)

            case "country":
                text = text.lower()
                if text in ["rus", "kaz", "blr"]:
                    return text

            case "gender":
                text_rus = pytesseract.image_to_string(
                    img, lang="rus", config=self.tesseract_config_rus)
                text_rus = text_rus.replace("\n\x0c", "")
                if ("M" in text) or ("М" in text_rus):
                    return Gender.MALE
                if ("F" in text) or ("Ж" in text_rus):
                    return Gender.FEMALE

            case id_ if id_ in ["birth_date", "issue_date", "expiry_date"]:
                text = re.sub(r"[^0-9.\s]", "", text).strip()
                text = text.replace(" .", ".")
                text = text.replace(". ", ".")
                text = text.replace(" ", ".")
                try:
                    date = datetime.strptime(text, "%d.%m.%Y")  # noqa: DTZ007
                except ValueError:
                    pass
                else:
                    now = datetime.now().date()  # noqa: DTZ005
                    year_diff = now.year - date.year
                    if (id_ == "birth_date" and (0 <= year_diff <= 100) or  # noqa: PLR2004
                            id_ == "issue_date" and (0 <= year_diff <= 20) or  # noqa: PLR2004
                                id_ == "expiry_date" and (-20 <= year_diff <= 20)):  # noqa: PLR2004
                                    return date

            case "birth_place":
                text_rus = pytesseract.image_to_string(
                    img, lang="rus", config=self.tesseract_config_rus)
                text_rus = text_rus.replace("\n\x0c", "")
                if ("/" in text) and ("/" in text_rus):
                    text_1 = text_rus.split("/")[0].strip()
                    text_2 = text.split("/")[-1].strip()
                    return f"{text_1} / {text_2}"

        return None


    def process_image(self, img: np.ndarray) -> tuple[PassportDetails, PassportImages]:
        grid_debug = ImageGrid(columns=3)
        grid_debug.add(img, "Original")

        corners, img_detect = self.detect_corners(img)
        grid_debug.add(img_detect, "Page detection")

        # Точки верхней и нижней страницы, полученные из KeypointsDetector
        # (в прямоугольниках нумеруем углы по часовой стрелке)
        corners_top = (corners[0], corners[1], corners[3], corners[2])
        corners_bottom = (corners[2], corners[3], corners[5], corners[4])

        scan, markers_top, markers_bottom \
            = self.adjust_pages_by_ref_frames(img, corners_top, corners_bottom)
        scan_contrast = self.apply_adoptive_contrast(scan)
        scan_contrast_sharp = self.apply_sharp_filter(scan_contrast)
        self.draw_markers(scan_contrast, [*markers_top, *markers_bottom])
        grid_debug.add(scan_contrast_sharp, "Scanned")

        rois = self.extract_rois(scan_contrast, draw=True)
        grid_debug.add(scan_contrast, "ROIs")

        grid_rois = ImageGrid(columns=1, row_height=60)
        for id_, roi in rois.items():
            grid_rois.add(roi, id_)
        grid_debug.add(grid_rois.render(), "Text detection")


        boxes = self.extract_text_boxes(rois, draw=True)
        grid_boxes = ImageGrid(columns=1, row_height=60)
        for id_, box in boxes.items():
            grid_boxes.add(box, id_)
        grid_debug.add(grid_boxes.render(), "Text boxes")

        details_dict = {}
        for id_, box in boxes.items():
            details_dict[id_] = self.recognize_field(id_, box)

        if None in (details_dict["surname"], details_dict["given_name"]):
            raise TextRecognitionError

        details = PassportDetails(**details_dict)

        photo = self.apply_adoptive_contrast(rois["photo"])
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        photo = self.apply_sharp_filter(photo)

        img_debug = grid_debug.render()

        images = PassportImages(
            scanned=scan_contrast_sharp,
            photo=photo,
            debug=img_debug,
        )

        return details, images
