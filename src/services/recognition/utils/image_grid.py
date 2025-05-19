import cv2
import numpy as np


class ImageGrid:
    def __init__(
            self,
            *,
            columns: int = 2,
            row_height: int = 420,
            header_height: int = 30,
            font_size: int = 15,
    ) -> None:
        self.stack = []
        self.columns = columns
        self.row_height = row_height
        self.header_height = header_height
        self.font_size = font_size

    def add(self, image: np.ndarray, title: str | None) -> None:
        self.stack.append((image, title))

    def render(self) -> np.ndarray:
        col_width = 0

        images = []
        titles = []

        for i in range(len(self.stack)):
            image = self.stack[i][0]
            scale = self.row_height / image.shape[0]
            width = int(np.ceil(image.shape[1] * scale))
            if width > col_width:
                col_width = width
            image = cv2.resize(image, (width, self.row_height))
            if len(image.shape) < 3:  # noqa: PLR2004
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            images.append(image)
            titles.append(self.stack[i][1])

        blank = np.zeros((self.header_height + self.row_height, col_width, 3), np.uint8)
        row_count = int(np.ceil(len(self.stack) / self.columns))

        rows = []
        for i in range(row_count):
            row = [np.copy(blank) for _ in range(self.columns)]
            _titles = titles[(i * self.columns): ((i + 1) * self.columns)]
            _images = images[(i * self.columns): ((i + 1) * self.columns)]
            for j in range(len(_images)):
                if _titles[j] is not None:
                    orig_font_size = 22
                    font_scale = self.font_size / orig_font_size
                    base_x = 5
                    base_y = int(self.header_height * 0.5 + self.font_size * 0.5)
                    cv2.putText(row[j], _titles[j], (base_x, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (255, 255, 255), 1)
                h, w = _images[j].shape[0], _images[j].shape[1]
                y1, y2 = self.header_height, self.header_height + h
                row[j][y1:y2, :w] = _images[j][:h, :w]
            rows.append(row)

        h_stacks = [np.hstack(row) for row in rows]
        if len(h_stacks) == 1:
            return h_stacks[0]
        return np.vstack(h_stacks)
