import re
from collections.abc import Sequence
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from numpy import ndarray
from passporteye import read_mrz
from svgelements import SVG

from entities.passport import Gender, PassportDetails, PassportImages, PassportType
from src.services.keypoints.keypoints_detector import KeypointsDetector
from src.services.keypoints.validate import validate_pages

from .exceptions import CornerDetectionError
from .utils.filters import clahe_filter, sharp_filter
from .utils.image_grid import ImageGrid
from .utils.ocr_helpers import detect_box
from .utils.shapes import Point, Rectangle
from .utils.skew_correction import adjust_corners


class PassportRecognizer:
    def __init__(self) -> None:

        self.mask_dir = Path(__file__).parent.joinpath("passport_mask/").resolve()

        self.keypoints_detector = KeypointsDetector()

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
            "mrz",
            "title",
            "photo",
        ]
        self.roi_upscale = 2

        self.tesseract_config_eng = \
            "--oem 3 --psm 7 -c tessedit_char_whitelist=' ./-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
        self.tesseract_config_rus = \
            "--oem 3 --psm 7 -c tessedit_char_whitelist=' ./-1234567890АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'"
        self.mrz_config_legacy = \
            "--oem 0 --tessdata-dir /usr/share/tessdata_legacy"


    def get_mask(self, passport_type: PassportType) -> SVG:
        mask_path = Path(self.mask_dir).joinpath(f"{passport_type}.svg")
        return SVG().parse(str(mask_path))


    @staticmethod
    def get_mask_frame(mask: SVG) -> Rectangle:
        frame_el = mask.get_element_by_id("frame")
        return Rectangle(frame_el.x, frame_el.y, frame_el.width, frame_el.height)


    def detect_corners(self, img: np.ndarray) -> tuple[list[Point], np.ndarray]:
        detected = self.keypoints_detector.predict_keypoints(img)
        orig_height, orig_width = img.shape[:2]
        if not validate_pages(detected, (orig_width, orig_height)):
            raise CornerDetectionError
        img_detect = img.copy()
        corners = [Point(d[0], d[1]) for d in detected]
        self.draw_markers(img_detect, corners)
        return corners, img_detect


    def adjust_pages_by_mask(
            self, img: np.ndarray,
            passport_type: PassportType,
            corners_top: Sequence[Point],
            corners_bottom: Sequence[Point]) -> tuple[ndarray, list[Point], list[Point]]:
        # Adjust by frames and crop images with margin
        mask = self.get_mask(passport_type)
        frame = self.get_mask_frame(mask)

        frame_top = Rectangle(frame.x, frame.y,
                              frame.width, frame.height / 2)
        page_top, corners_top = adjust_corners(
            img, corners_top, frame_top,
            margin=(self.scan_margin[1], self.scan_margin[0], 0, self.scan_margin[0]))

        frame_bottom = Rectangle(frame.x, frame.y + frame.height / 2,
                                 frame.width, frame.height / 2)
        page_bottom, corners_bottom = adjust_corners(
            img, corners_bottom, frame_bottom,
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


    def extract_rois(
            self, img: np.ndarray,
            passport_type: PassportType, *,
            draw: bool = False,
    ) -> dict[str, np.ndarray]:
        # Extract regions from source image
        img_copy = img.copy()
        mask = self.get_mask(passport_type)
        frame = self.get_mask_frame(mask)
        ims_roi = {}
        for id_ in self.roi_ids:
            if el := mask.get_element_by_id(id_):
                # Get rectangle from SVG mask
                rect = Rectangle(el.x - frame.x + self.scan_margin[0],
                                 el.y - frame.y + self.scan_margin[1],
                                 el.width, el.height)
                # Extract region
                (x1, y1), (x2, y2) = rect.p1.tup, rect.p3.tup
                ims_roi[id_] = np.copy(img_copy[y1:y2, x1:x2])
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
            if id_ == "mrz":
                box = Rectangle(0, 0, img.shape[1], img.shape[0])
            else:
                k = 0.2 if id_ == "title" else 0.3
                min_h = img.shape[0] * k
                box = detect_box(img, min_h)

            # Crop by box and apply upscale
            (x1, y1), (x2, y2) = box.p1.tup, box.p3.tup
            box_img = np.copy(img[y1:y2, x1:x2])
            resized = cv2.resize(box_img, None,
                                    fx=self.roi_upscale, fy=self.roi_upscale,
                                    interpolation=cv2.INTER_CUBIC)
            grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(grayed, None, h=7, templateWindowSize=5, searchWindowSize=21)
            boxes[id_] = denoised
            if draw:
                cv2.rectangle(img, box.p1.tup, box.p3.tup, (255, 0, 0), 1)
        return boxes


    def detect_passport_type(self, title_img: np.ndarray) -> PassportType:
        """Определяем тип паспорта по полю title."""
        text_rus = pytesseract.image_to_string(title_img, lang="rus", config=self.tesseract_config_rus)
        text_eng = pytesseract.image_to_string(title_img, lang="eng", config=self.tesseract_config_eng)
        if text_rus and "ФЕДЕРАЦИЯ" in text_rus.upper():
            if text_eng and "RUSSIAN" in text_eng.upper():
                return PassportType.RUS_10Y
            return PassportType.RUS_5Y
        return PassportType.OTHER


    def read_mrz_with_score(  # noqa: C901
            self, mrz_img: np.ndarray, *, legacy: bool = False) -> tuple[PassportDetails, int]:
        """Распознает и парсит MRZ с изображения. Работает как на фото целого паспорт, так и с вырезанным регионом."""
        # _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("MRZ", cleaned)
        # cv2.waitKey(0)

        _, buffer = cv2.imencode(".png", mrz_img)
        img_binary = BytesIO(buffer.tobytes())

        extra_cmd = ""
        if self.mrz_config_legacy and legacy:
            extra_cmd = self.mrz_config_legacy

        mrz = read_mrz(img_binary, extra_cmdline_params=extra_cmd)
        if mrz is None:
            return PassportDetails(), 0

        mrz_dict = mrz.to_dict()
        valid_score = mrz_dict["valid_score"]

        # Костыль. Боремся с лишними символами в имени при использовании tesseract 5.
        # Двойное имя в таком случае не определится, но мы можем этим пренебречь.
        if " " in mrz_dict["names"]:
            mrz_dict["names"] = mrz_dict["names"].split(" ")[0]

        # print("MRZ", legacy, mrz_dict)

        if "P" not in mrz_dict["type"]:
            return PassportDetails(), valid_score

        def mrz_passport_no(number_str: str) -> str | None:
            number_str = re.sub(r"[^0-9\s]", "", number_str)
            if len(number_str) != 9:
                return None
            return number_str

        def mrz_to_date(date_str: str) -> date | None:
            """Преобразует строку вида 'YYMMDD' в объект datetime.date."""
            if len(date_str) != 6 or not date_str.isdigit():
                return None
            try:
                return datetime.strptime(date_str, "%y%m%d").date()  # noqa: DTZ007
            except ValueError:
                return None

        def mrz_country(country_str: str) -> str | None:
            country_str = country_str.lower()
            if country_str in ["rus", "kaz", "blr"]:
                return country_str
            return None

        details_dict = {
            "passport_no": mrz_passport_no(mrz_dict["number"]),
            "surname": mrz_dict["surname"],
            "given_name": mrz_dict["names"],
            "country": mrz_country(mrz_dict["country"]),
            "nationality": mrz_country(mrz_dict["nationality"]),
            "birth_date": mrz_to_date(mrz_dict["date_of_birth"]),
            "expiry_date": mrz_to_date(mrz_dict["expiration_date"]),
        }
        if mrz_dict["sex"] and "M" in mrz_dict["sex"].upper():
            details_dict["gender"] = Gender.MALE
        if mrz_dict["sex"] and "F" in mrz_dict["sex"].upper():
            details_dict["gender"] = Gender.FEMALE

        return PassportDetails(**details_dict), valid_score


    def recognize_mrz(self, mrz_img: np.ndarray, passport_type: PassportType | None) -> PassportDetails:
        """Распознает и парсит MRZ с изображения. Выбирается лучший из нового и legacy вариантов."""
        details, score = self.read_mrz_with_score(mrz_img)
        details_dict = details.dict(exclude_none=True)
        details_legacy, score_legacy = self.read_mrz_with_score(mrz_img, legacy=True)
        details_dict_legacy = details_legacy.dict(exclude_none=True)
        # print(details)
        # print(details_dict)
        # print(details_legacy)
        # print(details_dict_legacy)
        if score_legacy > score:
            details_dict |= details_dict_legacy
        else:
            details_dict = details_dict_legacy | details_dict
        result = PassportDetails(**details_dict)
        if result.expiry_date:
            if passport_type == PassportType.RUS_10Y:
                result.issue_date = result.expiry_date.replace(year=result.expiry_date.year - 10)
            elif passport_type == PassportType.RUS_5Y:
                result.issue_date = result.expiry_date.replace(year=result.expiry_date.year - 5)
        return result


    def recognize_field(self, id_: str, img: np.ndarray) -> Any:
        # Извлечение текста на английском
        text = pytesseract.image_to_string(img, lang="eng", config=self.tesseract_config_eng)
        text = text.replace("\n\x0c", "").replace("\n", "")

        # Обработка полей с именем и фамилией
        if id_ in ["surname", "given_name"]:
            return re.sub(r"[^A-Z]", "", text)

        # Обработка номера паспорта
        if id_ == "passport_no":
            text = re.sub(r"[^0-9\s]", "", text)
            if len(text) != 9:
                return None
            return text

        # Обработка кода страны
        if id_ == "country":
            text = text.lower()
            if text in ["rus", "kaz", "blr"]:
                return text

        # Обработка пола
        if id_ == "gender":
            text_rus = pytesseract.image_to_string(img, lang="rus", config=self.tesseract_config_rus)
            text_rus = text_rus.replace("\n\x0c", "").replace("\n", "")
            if "M" in text or "М" in text_rus:
                return Gender.MALE
            if "F" in text or "Ж" in text_rus:
                return Gender.FEMALE

        # Обработка дат
        if id_ in ["birth_date", "issue_date", "expiry_date"]:
            text = re.sub(r"[^0-9.\s]", "", text).strip().replace(" .", ".").replace(". ", ".").replace(" ", ".")
            try:
                date = datetime.strptime(text, "%d.%m.%Y").date()  # noqa: DTZ007
                now = datetime.now().date()  # noqa: DTZ005
                year_diff = now.year - date.year
                if (
                        (id_ == "birth_date" and 0 <= year_diff <= 100) or  # noqa: PLR2004
                        (id_ == "issue_date" and 0 <= year_diff <= 20) or  # noqa: PLR2004
                        (id_ == "expiry_date" and -20 <= year_diff <= 20)  # noqa: PLR2004
                ):
                    return date
            except ValueError:
                pass

        # Обработка места рождения
        if id_ == "birth_place":
            text_rus = pytesseract.image_to_string(img, lang="rus", config=self.tesseract_config_rus)
            text_rus = text_rus.replace("\n\x0c", "").replace("\n", "")
            if "/" in text and "/" in text_rus:
                return f"{text_rus.split('/')[0].strip()} / {text.split('/')[-1].strip()}"

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
            = self.adjust_pages_by_mask(img, PassportType.RUS_10Y, corners_top, corners_bottom)
        scan_contrast = self.apply_adoptive_contrast(scan)
        scan_contrast_sharp = self.apply_sharp_filter(scan_contrast)
        rois = self.extract_rois(scan_contrast, PassportType.RUS_10Y, draw=True)
        self.draw_markers(scan_contrast, [*markers_top, *markers_bottom])
        boxes = self.extract_text_boxes(rois, draw=True)
        passport_type = self.detect_passport_type(boxes["title"])
        # print("passport_type", passport_type)

        if passport_type not in (PassportType.RUS_10Y, PassportType.OTHER):
            scan, markers_top, markers_bottom \
                = self.adjust_pages_by_mask(img, passport_type, corners_top, corners_bottom)
            scan_contrast = self.apply_adoptive_contrast(scan)
            scan_contrast_sharp = self.apply_sharp_filter(scan_contrast)
            rois = self.extract_rois(scan_contrast, passport_type, draw=True)
            self.draw_markers(scan_contrast, [*markers_top, *markers_bottom])
            boxes = self.extract_text_boxes(rois, draw=True)


        grid_debug.add(scan_contrast_sharp, "Scanned")
        grid_debug.add(scan_contrast, "ROIs")
        details_dict = {}

        if passport_type != PassportType.OTHER:
            grid_rois = ImageGrid(columns=1, row_height=60)
            for id_, roi in rois.items():
                grid_rois.add(roi, id_)
            grid_debug.add(grid_rois.render(), "Text detection")

            grid_boxes = ImageGrid(columns=1, row_height=60)
            for id_, box in boxes.items():
                grid_boxes.add(box, id_)
            grid_debug.add(grid_boxes.render(), "Text boxes")

            for id_, box in boxes.items():
                details_dict[id_] = self.recognize_field(id_, box)

        # print("Details from fields:", details_dict)
        # if None in (details_dict["surname"], details_dict["given_name"]):
        #     raise TextRecognitionError

        details_mrz = self.recognize_mrz(boxes["mrz"], passport_type)
        details_mrz_dict = details_mrz.dict(exclude_none=True)
        # print("Details from MRZ:", details_mrz_dict)

        details_dict |= details_mrz_dict
        details = PassportDetails(**details_dict)
        # print("Details:", details)

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
