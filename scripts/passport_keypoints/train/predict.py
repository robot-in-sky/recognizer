import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
from model import KeypointRegressor0


@dataclass
class PredictConfig:
    model_path: Path
    target_size: tuple[int, int]
    source_dir: Path
    extensions: list[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_json(cls, path: Path) -> "PredictConfig":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model_path=Path(data["model_path"]),
            target_size=(int(data["target_size"][0]), int(data["target_size"][1])),
            source_dir=Path(data["source_dir"]),
            extensions=data["extensions"],
            device=data.get("device", cls.device),
        )


def load_model(config: PredictConfig) -> torch.nn.Module:
    model = KeypointRegressor0()
    state_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def preprocess(image: np.ndarray, target_size: tuple[int, int]) -> tuple[torch.Tensor, np.ndarray]:
    original = image.copy()
    resized = cv2.resize(image, (target_size[1], target_size[0]))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor, original


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    image = image.copy()
    for i, (x, y) in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(image, str(i + 1), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return image


def is_passport_detected(keypoints: np.ndarray, image_shape: tuple[int, int]) -> bool:
    h, w = image_shape[:2]
    if not np.all((keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) &
                  (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)):
        return False
    dists = [np.linalg.norm(a - b) for a, b in combinations(keypoints, 2)]
    avg_dist = np.mean(dists)
    return avg_dist > min(h, w) * 0.05


def get_image_paths(config: PredictConfig) -> list[Path]:
    return sorted([
        p for ext in config.extensions
        for p in config.source_dir.glob(f"*.{ext.lower()}")
    ])


def run(config: PredictConfig) -> None:
    model = load_model(config)
    image_paths = get_image_paths(config)

    if not image_paths:
        print("No images found in", config.source_dir)
        return

    index = 0
    while True:
        print(f"Processing image: {image_paths[index].name}")
        image = cv2.imread(str(image_paths[index]))
        if image is None:
            print(f"Failed to load {image_paths[index]}")
            continue

        input_tensor, original = preprocess(image, config.target_size)
        input_tensor = input_tensor.to(config.device)

        with torch.no_grad():
            output = model(input_tensor).cpu().numpy().reshape(-1, 2)
            height, width = original.shape[:2]
            output *= [width, height]

        passport_found = is_passport_detected(output, original.shape)
        label = "Passport detected" if passport_found else "No passport"

        vis = draw_keypoints(original, output)
        cv2.putText(vis, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0) if passport_found else (0, 0, 255), 2)

        cv2.imshow("Detection", vis)
        key = cv2.waitKey(0)

        if key == 27:  # ESC  # noqa: PLR2004
            break
        if key == 81:  # Left arrow  # noqa: PLR2004
            index = (index - 1) % len(image_paths)
        elif key == 83:  # Right arrow  # noqa: PLR2004
            index = (index + 1) % len(image_paths)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_config = PredictConfig.from_json(Path("predict.json"))
    run(run_config)
