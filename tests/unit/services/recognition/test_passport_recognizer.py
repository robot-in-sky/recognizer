from pathlib import Path

import cv2
import pytest
import yaml
from src.entities.passport import PassportDetails
from src.services.recognition import PassportRecognizer
from src.services.recognition.exceptions import CornerDetectionError, ImageMismatchError


def count_mismatches(str_value: str, expected_value: str) -> int:
    return (sum(1 for v, e in zip(str_value, expected_value, strict=False) if v != e) +
            abs(len(str_value) - len(expected_value)))


class TestPassportRecognizer:

    @pytest.mark.parametrize("src_basename", [
        "1",
        "2",
        "3",
        "4",
        "5",
        "7",
        "8",
    ])
    async def test_check_details(self, src_basename: str) -> None:
        input_path = Path("input")
        src_path = input_path.joinpath(f"{src_basename}.jpg")
        details_path = input_path.joinpath(f"{src_basename}.yml")
        src_img = cv2.imread(str(src_path))
        recognizer = PassportRecognizer()
        details, images = recognizer.process_image(src_img)
        with Path.open(details_path, encoding="utf-8") as file:
            details_dict = yaml.safe_load(file)
            expected = PassportDetails(**details_dict)

        assert count_mismatches(details.given_name, expected.given_name) <= 1
        assert count_mismatches(details.surname, expected.surname) <= 1
        assert count_mismatches(details.passport_no, expected.passport_no) <= 1
        assert details.country == expected.country
        assert count_mismatches(str(details.birth_date), str(expected.birth_date)) <= 1
        assert details.gender == expected.gender
        # assert count_mismatches(details.birth_place, expected.birth_place) <= 1
        # assert count_mismatches(str(details.issue_date), str(expected.issue_date)) <= 1
        # assert count_mismatches(str(details.expiry_date), str(expected.expiry_date)) <= 1


    @pytest.mark.parametrize("src_basename", [
        "c1",
        "c2",
    ])
    async def test_corner_detection_exception(self, src_basename: str) -> None:
        input_path = Path("input")
        src_path = input_path.joinpath(f"{src_basename}.jpg")
        src_img = cv2.imread(str(src_path))
        recognizer = PassportRecognizer()
        with pytest.raises(CornerDetectionError):
            _, _ = recognizer.process_image(src_img)


    @pytest.mark.parametrize("src_basename", [
        "m1",
        "m2",
    ])
    async def test_image_mismatch_exception(self, src_basename: str) -> None:
        input_path = Path("input")
        src_path = input_path.joinpath(f"{src_basename}.jpg")
        src_img = cv2.imread(str(src_path))
        recognizer = PassportRecognizer()
        with pytest.raises(ImageMismatchError):
            _, _ = recognizer.process_image(src_img)


