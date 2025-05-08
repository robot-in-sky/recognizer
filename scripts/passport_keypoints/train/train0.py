import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from dataset import PassportKeypointDataset, split_indices
from model import KeypointRegressor0
from torch import nn
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


def train(config_path: Path = Path("train.json")) -> None:
    config = TrainConfig.from_json(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(config)
    model = KeypointRegressor0(num_points=6).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["keypoints"].to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

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