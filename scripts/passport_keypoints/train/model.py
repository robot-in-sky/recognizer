import timm
from torch import Tensor, nn


class KeypointRegressor(nn.Module):
    def __init__(self, num_points: int = 6) -> None:
        super().__init__()
        # Загружаем предобученную EfficientNet-B0 без классификатора
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="")
        # Добавляем свёрточный слой для предсказания 6 тепловых карт
        self.head = nn.Conv2d(1280, num_points, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Пропускаем через backbone (без глобального пулинга)
        x = self.backbone.forward_features(x)  # Выход: [batch_size, 1280, H, W]
        # Применяем свёрточный слой для получения тепловых карт
        x = self.head(x)  # Выход: [batch_size, num_points, H, W]
        return x  # noqa: RET504


class KeypointRegressor0(nn.Module):
    def __init__(self, num_points: int = 6) -> None:
        super().__init__()
        # Загружаем предобученную EfficientNet-B0
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        # Заменяем последний слой (classifier) на линейный для регрессии координат
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_points * 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)