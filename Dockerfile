FROM python:3.12-slim-bullseye

ENV POETRY_VERSION=1.7.1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    APP_DIR="/app"
ENV PATH="$POETRY_HOME/bin:$PATH" \
    PYTHONPATH="$APP_DIR:$APP_DIR/src"

RUN python -m pip install --upgrade pip
RUN pip install "poetry==${POETRY_VERSION}"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 libglib2.0-0 \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Загрузка legacy Tesseract-моделей
RUN mkdir -p /usr/share/tessdata_legacy && \
    wget -O /usr/share/tessdata_legacy/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata && \
    wget -O /usr/share/tessdata_legacy/rus.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata

# Установка OpenCV без GUI
RUN pip install --no-cache-dir opencv-python-headless

# Установка CPU-версий torch и torchvision
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR $APP_DIR
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip install --no-cache-dir -r requirements.txt
RUN rm pyproject.toml poetry.lock

COPY src/ ./src

CMD ["python", "src/main.py"]
