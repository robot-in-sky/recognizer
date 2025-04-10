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

from src.entities.passport import Gender, PassportDetails, PassportImages

from .exceptions import CornerDetectionError, ImageMismatchError, TextRecognitionError
from .utils.filters import clahe_filter, sharp_filter
from .utils.image_grid import ImageGrid
from .utils.ocr_helpers import detect_box
from .utils.shapes import Point, Rectangle
from .utils.skew_correction import adjust_corners, correct_skew, detect_corners


class PassportRecognizer:
    def __init__(self) -> None:

        ref_dir = Path(__file__).parent.joinpath("ref/").resolve()
        ref_img_path = Path(ref_dir).joinpath("passport.png")
        ref_mask_path = Path(ref_dir).joinpath("passport.svg")

        self._ref_img = cv2.imread(str(ref_img_path))
        self._ref_mask = SVG().parse(str(ref_mask_path))

        frame_el = self._ref_mask.get_element_by_id("frame")
        self._frame = Rectangle(frame_el.x, frame_el.y, frame_el.width, frame_el.height)
        self._frame_top = Rectangle(self._frame.x, self._frame.y,
                                    self._frame.width, self._frame.height / 2)
        self._frame_bottom = Rectangle(self._frame.x, self._frame.y + self._frame.height / 2,
                                        self._frame.width, self._frame.height / 2)

        self.scan_margin = (20, 20)

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


    def scale_to_ref(self, img: np.ndarray) -> np.ndarray:
        resize_width = self._ref_img.shape[1]
        height, width = img.shape[:2]
        resize_height = round(resize_width * height / width)
        if width == resize_width:
            return img.copy()
        return cv2.resize(img, (resize_width, resize_height))


    def correct_skew_by_ref_features(self, img: np.ndarray) -> np.ndarray:
        try:
            return correct_skew(img, self._ref_img)
        except ValueError:
            raise ImageMismatchError from None


    def detect_top_and_bottom_page_corners(
            self, img: np.ndarray) -> tuple[list[Point], list[Point], np.ndarray, np.ndarray]:

        detection_width = 40
        edge_min_length = 80

        # Detect top page corners
        corners_top, img_detect_top = detect_corners( img,
                                        roi_centerline=self._frame_top,
                                        roi_bandwidth=detection_width,
                                        edge_min_length=edge_min_length)

        # Detect bottom page corners
        corners_bottom, img_detect_bottom = detect_corners( img,
                                                roi_centerline=self._frame_bottom,
                                                roi_bandwidth=detection_width,
                                                edge_min_length=edge_min_length)

        # Common fold line
        corners_top[2] = corners_bottom[1] or corners_top[2]
        corners_top[3] = corners_bottom[0] or corners_top[3]
        corners_bottom[1] = corners_top[2] or corners_bottom[1]
        corners_bottom[0] = corners_top[3] or corners_bottom[0]

        # Refined external corner detection if fold line is detected
        if None not in (corners_top[2], corners_top[3]):
            fold_length = corners_top[2].x - corners_top[3].x
            _frame_height = self._frame.height / 2 + detection_width / 2

            detection_width = 70
            edge_min_length = 140

            if None in (corners_top[0], corners_top[1]):

                # Refined top corners detection
                _frame_top = Rectangle(corners_top[3].x, corners_top[3].y - _frame_height,
                                        fold_length, _frame_height)
                _corners_top, img_detect_top = detect_corners(img,
                                                 roi_centerline=_frame_top,
                                                 roi_bandwidth=detection_width,
                                                 edge_min_length=edge_min_length)
                corners_top[0] = corners_top[0] or _corners_top[0]
                corners_top[1] = corners_top[1] or _corners_top[1]

            if None in (corners_bottom[2], corners_bottom[3]):

                # Refined bottom corners detection
                _frame_bottom = Rectangle(corners_top[3].x, corners_top[3].y,
                                       fold_length, _frame_height)
                _corners_bottom, img_detect_bottom = detect_corners(img,
                                                    roi_centerline=_frame_bottom,
                                                    roi_bandwidth=detection_width,
                                                    edge_min_length=edge_min_length)
                corners_bottom[2] = corners_bottom[2] or _corners_bottom[2]
                corners_bottom[3] = corners_bottom[3] or _corners_bottom[3]

        return corners_top, corners_bottom, img_detect_top, img_detect_bottom


    def adjust_pages_by_ref_frames(
            self, img: np.ndarray,
            corners_top: Sequence[Point],
            corners_bottom: Sequence[Point]) -> tuple[ndarray, list[Point], list[Point]]:
        # Adjust by frames and crop images with margin
        page_top, markers_top = adjust_corners(
            img, corners_top, self._frame_top,
            margin=(self.scan_margin[1], self.scan_margin[0], 0, self.scan_margin[0]))

        page_bottom, markers_bottom = adjust_corners(
            img, corners_bottom, self._frame_bottom,
            margin=(0, self.scan_margin[0], self.scan_margin[1], self.scan_margin[0]))

        for i in range(len(markers_bottom)):
            markers_bottom[i].y += page_top.shape[0]

        # Stack top and bottom pages
        return np.vstack((page_top, page_bottom)), markers_top, markers_bottom


    @staticmethod
    def draw_markers(
            img: np.ndarray,
            markers_top: Sequence[Point],
            markers_bottom: Sequence[Point]) -> None:
        pts = [*markers_top, *markers_bottom]
        for p in pts:
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


    def recognize_field(self, id_: str, img: np.ndarray) -> Any:  # noqa: PLR0911, C901, PLR0912
        text = pytesseract.image_to_string(
            img, lang="eng", config=self.tesseract_config_eng)
        text = text.replace("\n\x0c", "")

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
        grid_debug = ImageGrid()

        img = self.scale_to_ref(img)
        grid_debug.add(img, "Original")

        img_corrected = self.correct_skew_by_ref_features(img)
        grid_debug.add(img_corrected, "Corrected")

        corners_top, corners_bottom, img_detect_top, img_detect_bottom \
                    = self.detect_top_and_bottom_page_corners(img_corrected)
        grid_debug.add(img_detect_top, "Top page detection")
        grid_debug.add(img_detect_bottom, "Bottom page detection")

        if None in corners_top or None in corners_bottom:
            raise CornerDetectionError

        scan, markers_top, markers_bottom \
            = self.adjust_pages_by_ref_frames(img_corrected, corners_top, corners_bottom)
        scan_contrast = self.apply_adoptive_contrast(scan)
        scan_contrast_sharp = self.apply_sharp_filter(scan_contrast)
        self.draw_markers(scan_contrast, markers_top, markers_bottom)
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

        images = PassportImages(
            scanned=scan_contrast_sharp,
            photo=photo,
            debug=grid_debug.render(),
        )

        return details, images
