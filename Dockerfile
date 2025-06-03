FROM python:3.12-slim-bullseye

ENV POETRY_VERSION=1.7.1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    APP_DIR="/app"
ENV PATH="$POETRY_HOME/bin:$PATH" \
    PYTHONPATH=${APP_DIR}

RUN python -m pip install --upgrade pip
RUN pip install "poetry==${POETRY_VERSION}"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Загрузка Tesseract-моделей
RUN mkdir -p /usr/share/tessdata_legacy && \
    wget -O /usr/share/tessdata/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata && \
    wget -O /usr/share/tessdata/rus.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata && \
    wget -O /usr/share/tessdata_legacy/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata && \
    wget -O /usr/share/tessdata_legacy/rus.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata

# Установка OpenCV без GUI
RUN pip install --no-cache-dir opencv-python-headless

WORKDIR $APP_DIR
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip install --no-cache-dir -r requirements.txt
RUN rm pyproject.toml poetry.lock

COPY src/ ./src

CMD ["python", "src/main.py"]
