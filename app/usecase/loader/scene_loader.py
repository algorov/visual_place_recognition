import csv
import os
from typing import Dict, List, Any
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def load_scene_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Функция загружает метаданные сцен из CSV-файла.

    Она читает файл по указанному пути и формирует словарь, где ключом выступает
    идентификатор сцены (scene_id), а значением — словарь с названием, описанием,
    широтой и долготой сцены.

    При некорректных координатах функция выводит предупреждение и устанавливает
    значения координат в 0.0.
    """
    metadata = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            scene_id = row.get('scene_id')
            if not scene_id:
                # Если идентификатор сцены отсутствует, соответствующая запись пропускается.
                continue

            try:
                lat, lon = float(row.get('lat', 0.0)), float(row.get('lon', 0.0))
            except ValueError:
                # В случае ошибки преобразования координат выводится предупреждение,
                # и координаты устанавливаются в 0.0 по умолчанию.
                print(f"⚠️ Неверные координаты в сцене {scene_id}, устанавливаем 0.0")
                lat, lon = 0.0, 0.0

            # Формируется запись с метаданными для каждой сцены.
            metadata[scene_id] = {
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'lat': lat,
                'lon': lon,
            }

    return metadata


def load_scene_dataset(dataset_path: str, metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Эта функция собирает список изображений и их метаданных из заданной папки с датасетом.

    Для каждого каталога (предполагаемого как отдельная сцена) она извлекает
    соответствующие метаданные из переданного словаря и добавляет в итоговый список
    все изображения с поддерживаемыми расширениями.

    Итоговый список содержит словари с идентификатором сцены, названием, описанием,
    абсолютным путем к изображению, а также координатами.
    """
    entries = []
    base_path = Path(dataset_path)

    for scene_entry in base_path.iterdir():
        if not scene_entry.is_dir():
            # Пропускаются элементы, которые не являются папками.
            continue

        scene_id = scene_entry.name

        # Получение метаданных сцены или использование значений по умолчанию.
        meta = metadata.get(scene_id, {})
        lat = meta.get('lat', 0.0)
        lon = meta.get('lon', 0.0)
        title = meta.get('title', '')
        description = meta.get('description', '')

        for file_entry in scene_entry.iterdir():
            if file_entry.is_file() and file_entry.suffix.lower() in VALID_EXTENSIONS:
                # Для каждого подходящего файла формируется запись с данными.
                entries.append({
                    'scene_id': scene_id,
                    'title': title,
                    'description': description,
                    'path': str(file_entry.resolve()),
                    'lat': lat,
                    'lon': lon,
                })

    return entries

def load_scene_images_by_id(scene_id: str, base_dir: str = "data/scenes/") -> List[Image.Image]:
    """
    Загружает все изображения сцены по scene_id из директории.

    Args:
        scene_id (str): ID сцены, например "scene_5"
        base_dir (str): Базовая директория с изображениями сцен

    Returns:
        List[PIL.Image.Image]: список изображений
    """
    scene_dir = os.path.join(base_dir, scene_id)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Директория для scene_id={scene_id} не найдена: {scene_dir}")

    images = []
    for filename in os.listdir(scene_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(scene_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"⚠️ Не удалось загрузить {img_path}: {e}")

    if not images:
        raise FileNotFoundError(f"Не найдено изображений в директории: {scene_dir}")

    return images
