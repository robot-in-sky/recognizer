from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def letterbox_resize(image: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Масштабирует изображение с сохранением пропорций и добавляет паддинги."""
    orig_height, orig_width = image.shape[:2]
    target_height, target_width = target_size
    # Вычисляем масштаб
    scale = min(target_width / orig_width, target_height / orig_height)
    new_width, new_height = int(orig_width * scale), int(orig_height * scale)
    # Масштабируем изображение
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Создаем пустое изображение с целевым размером
    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # Вычисляем паддинги для центрирования
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2
    # Копируем масштабированное изображение в центр
    padded[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized
    return padded, scale, (pad_top, pad_left)


def rotate_image(image: np.ndarray,
                     angle: float, *, scale: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=scale)
    # Выполняем поворот
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def rotate_points(points: np.ndarray,
                      image_shape: tuple[int, int],
                      angle: float, *, scale: float = 1.0) -> np.ndarray:
    h, w = image_shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=scale)
    # Преобразуем в абсолютные координаты
    abs_points = points.copy()
    abs_points[:, 0] *= w
    abs_points[:, 1] *= h
    # Добавим фиктивную 1 для смещения
    ones = np.ones((abs_points.shape[0], 1), dtype=np.float32)
    points_aug = np.hstack([abs_points, ones])
    # Применим матрицу поворота
    rotated = np.dot(points_aug, rotation_matrix.T)
    # Обратно в нормализованные координаты
    rotated[:, 0] /= w
    rotated[:, 1] /= h
    return rotated


class PassportKeypointDataset(Dataset):

    def __init__(  # noqa: PLR0913
            self,
            images_dir: Path,
            target_size: tuple[int, int],
            annotations_path: Path, *,
            indices: Sequence[int] | None = None,
            augment: bool = True,
    ) -> None:

        self.images_dir = images_dir
        self.target_size = target_size
        self.annotations_path = annotations_path
        self.augment = augment

        with self.annotations_path.open("r", encoding="utf-8") as f:
            original_annotations = yaml.safe_load(f) or []

        # Создаем аннотации для оригинала и 5 повернутых версий
        self.annotations = []
        rotation_angles = [0, 5, 15, 165, 175, 180, 185, 195, 345, 355]  # 0 - оригинал, остальные - повороты
        for ann in original_annotations:
            for angle in rotation_angles:
                new_ann = ann.copy()
                new_ann["rotation"] = angle
                new_ann["source"] = ann["image_name"]  # Оригинальное имя файла
                self.annotations.append(new_ann)

        if indices is not None:
            self.annotations = [self.annotations[i] for i in indices]

        # Базовая предобработка
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Случайные аугментации (без поворотов)
        if self.augment:
            self.aug_pipeline = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(contrast=0.2),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ], p=0.2),
            ])
        else:
            self.aug_pipeline = transforms.Compose([])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        item = self.annotations[idx]
        image_path = self.images_dir / Path(item["source"]).name

        # Загружаем оригинальное изображение
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Применяем letterbox resize
        image, scale, (pad_top, pad_left) = letterbox_resize(image, self.target_size)

        # Нормализуем пиксельные координаты с учетом масштаба и паддингов
        points = np.array(item["points"], dtype=np.float32)  # Пиксельные координаты оригинала
        points[:, 0] = (points[:, 0] * scale + pad_left) / self.target_size[1]  # Нормализация x
        points[:, 1] = (points[:, 1] * scale + pad_top) / self.target_size[0]  # Нормализация y
        points = np.clip(points, 0, 1)  # Ограничение [0, 1]

        # Применяем фиксированный поворот из аннотации
        rotation = item.get("rotation", 0.0)
        if rotation != 0.0:
            scale = 0.8 if rotation != 180 else 1.0  # noqa: PLR2004
            image = rotate_image(image, rotation, scale=scale)
            points = rotate_points(points, self.target_size, rotation, scale=scale)

        # Конвертируем в PIL для torchvision аугментаций
        image = Image.fromarray(image)

        if self.augment:
            # Применяем случайные аугментации (яркость, контраст, размытие)
            image = self.aug_pipeline(image)

        image = self.preprocess(image)  # [C, H, W] float32 tensor
        keypoints = torch.tensor(points.flatten(), dtype=torch.float32)

        return {
            "image": image,
            "keypoints": keypoints,
            "source": item["source"],
            "rotation": rotation,
        }


def split_indices(n_total: int, val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng()
    indices = rng.permutation(n_total).tolist()
    split = int(n_total * (1 - val_ratio))
    return indices[:split], indices[split:]
