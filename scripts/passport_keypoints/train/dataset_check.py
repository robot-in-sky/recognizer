import shutil
from pathlib import Path

import matplotlib as mpl
import yaml
from pydantic import BaseModel, Field

from scripts.passport_keypoints.train.dataset import PassportKeypointDataset

mpl.use("Agg")  # Устанавливаем бэкенд Agg (неинтерактивный)
import matplotlib.pyplot as plt


class TargetSizeConfig(BaseModel):
    width: int = Field(..., gt=0, description="Width of target image")
    height: int = Field(..., gt=0, description="Height of target image")


class DatasetConfig(BaseModel):
    images_dir: Path = Field(..., description="Directory with source images")
    annotations_path: Path = Field(..., description="Path to annotations YML file")
    target_size: TargetSizeConfig = Field(..., description="Target image size (width, height)")
    augment: bool = Field(..., description="Whether to apply augmentations")


class CheckSettings(BaseModel):
    output_dir: Path = Field(..., description="Directory to save checked images")
    max_files: int = Field(..., gt=0, description="Maximum number of files to save")


class DatasetCheckConfig(BaseModel):
    dataset: DatasetConfig
    check: CheckSettings

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DatasetCheckConfig":
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def save_data_to_check(
    dataset: PassportKeypointDataset,
    output_dir: Path,
    max_files: int = 60,
) -> None:
    # Сохраняет изображения из датасета с отмеченными ключевыми точками.
    # Имена файлов формируются на основе поля source и rotation (например, 11_rot180.jpg).

    # Создаем выходную папку, удаляя старую, если она существует
    check_dir = output_dir / "dataset_check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)

    # Ограничиваем количество сохраняемых файлов
    num_files = min(max_files, len(dataset))

    # Размеры целевого изображения
    target_height, target_width = dataset.target_size  # (640, 480)

    for i in range(num_files):
        # Получаем элемент датасета
        item = dataset[i]
        image = item["image"]  # [C, H, W]
        keypoints = item["keypoints"]  # [num_points * 2]
        source = item["source"]  # Оригинальное имя, например, 11.jpg
        rotation = item["rotation"]  # Угол поворота, например, 180.0

        # Формируем имя файла
        if rotation == 0.0:
            output_filename = source
        else:
            base_name = Path(source).stem
            suffix = Path(source).suffix
            output_filename = f"{base_name}_rot{int(rotation)}{suffix}"

        # Преобразуем изображение в numpy для отображения
        image_np = image.permute(1, 2, 0).numpy()  # [H, W, C], нормализовано [0, 1]

        # Преобразуем ключевые точки в пиксельные координаты
        keypoints = keypoints.reshape(-1, 2).numpy()  # [num_points, 2]
        keypoints[:, 0] = keypoints[:, 0] * target_width  # x * 480
        keypoints[:, 1] = keypoints[:, 1] * target_height  # y * 640

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(target_width / 100, target_height / 100), dpi=100)
        ax.imshow(image_np)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c="red", s=50)
        ax.set_title(f"Normalized: {output_filename}")
        ax.axis("off")

        # Сохраняем изображение
        output_path = check_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"Saved {output_filename}")


if __name__ == "__main__":
    # Загружаем конфигурацию
    config = DatasetCheckConfig.from_yaml(Path("dataset_check.yml"))

    # Создаем датасет с параметрами из конфигурации
    _dataset = PassportKeypointDataset(
        images_dir=config.dataset.images_dir,
        target_size=(config.dataset.target_size.height, config.dataset.target_size.width),
        annotations_path=config.dataset.annotations_path,
        augment=config.dataset.augment,
    )

    # Вызываем функцию проверки
    save_data_to_check(
        _dataset,
        output_dir=config.check.output_dir,
        max_files=config.check.max_files,
    )
