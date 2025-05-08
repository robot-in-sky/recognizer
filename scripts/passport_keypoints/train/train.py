import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dataset import PassportKeypointDataset, split_indices
from model import KeypointRegressor
from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    images_dir: Path
    target_size: tuple[int, int]
    annotations_path: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    augment: bool = True
    val_ratio: float = 0.0

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            images_dir=Path(data["images_dir"]),
            target_size=(int(data["target_size"][0]), int(data["target_size"][1])),
            annotations_path=Path(data["annotations_path"]),
            output_dir=Path(data["output_dir"]),
            epochs=data["epochs"],
            batch_size=data["batch_size"],
            learning_rate=data["learning_rate"],
            augment=data.get("augment", cls.augment),
            val_ratio=data.get("val_ratio", cls.val_ratio),
        )


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
    dataset = PassportKeypointDataset(
        images_dir=config.images_dir,
        target_size=config.target_size,
        annotations_path=config.annotations_path,
        augment=config.augment,
    )
    if config.val_ratio > 0:
        train_idx, val_idx = split_indices(len(dataset), val_ratio=config.val_ratio)
        train_set = PassportKeypointDataset(
            images_dir=config.images_dir,
            target_size=config.target_size,
            annotations_path=config.annotations_path,
            augment=config.augment,
            indices=train_idx,
        )
        val_set = PassportKeypointDataset(
            images_dir=config.images_dir,
            target_size=config.target_size,
            annotations_path=config.annotations_path,
            augment=False,
            indices=val_idx,
        )
        return (
            DataLoader(train_set, batch_size=config.batch_size, shuffle=True),
            DataLoader(val_set, batch_size=config.batch_size, shuffle=False),
        )

    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True), None


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["keypoints"].to(device)
            predictions = model(images)
            total_loss += criterion(predictions, targets).item()
    model.train()
    return total_loss / len(val_loader)


def save_model(model: nn.Module, output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    output_path = output_dir / f"keypoint_model_{timestamp}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def generate_heatmap(h, w, keypoints, sigma=1, input_image_size=224):
    keypoints = torch.tensor(keypoints).flatten() if isinstance(keypoints, (list, np.ndarray)) else keypoints.flatten()
    if len(keypoints) != 12:
        raise ValueError(f"Expected 12 coordinates (6 points), got {len(keypoints)}")
    keypoints = keypoints.reshape(-1, 2).tolist()

    heatmaps = np.zeros((6, h, w), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        x = x * w / input_image_size
        y = y * h / input_image_size
        for r in range(h):
            for c in range(w):
                heatmaps[i, r, c] = np.exp(-((r - y) ** 2 + (c - x) ** 2) / (2 * sigma ** 2))
    return torch.tensor(heatmaps)


def train(config_path: Path = Path("train.json")) -> None:
    config = TrainConfig.from_json(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(config)
    model = KeypointRegressor(num_points=6).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)  # [batch_size, 3, 224, 224]
            keypoints = batch["keypoints"]  # [batch_size, 12]

            # Проверяем формат keypoints
            if keypoints.shape != (config.batch_size, 12):
                raise ValueError(f"Expected keypoints shape [batch_size, 12], got {keypoints.shape}")

            # Отладка: проверяем, что передаём в generate_heatmap
            for kp in keypoints:
                print(f"Keypoints shape: {kp.shape}, values: {kp.tolist()}")
                break  # Проверяем только первый элемент

            # Генерируем тепловые карты
            target = torch.stack([
                generate_heatmap(7, 7, kp.tolist(), sigma=1, input_image_size=224)
                for kp in keypoints
            ]).to(device)  # [batch_size, 6, 7, 7]

            pred = model(images)  # [batch_size, 6, 7, 7]
            print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")  # Отладка
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.6f}")

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")

    save_model(model, config.output_dir)


if __name__ == "__main__":
    train()
