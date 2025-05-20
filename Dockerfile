# Базовый образ Python
FROM python:3.9-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \  # Для OpenCV
    libsm6 \           # Для GUI-библиотек
    libxext6

WORKDIR /app

# Копирование зависимостей и установка
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Запуск сервиса
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]