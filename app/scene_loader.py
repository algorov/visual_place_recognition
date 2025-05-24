import os
import csv
from typing import Dict, List, Any


def load_scene_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Загружает метаданные сцен из CSV в словарь.
    Ключ — scene_id, значение — словарь с title, description, lat, lon.
    """
    metadata = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row.get('scene_id')
            if not scene_id:
                continue  # пропускаем строки без scene_id

            try:
                lat = float(row.get('lat', 0.0))
                lon = float(row.get('lon', 0.0))
            except ValueError:
                lat, lon = 0.0, 0.0  # если конвертация не удалась

            metadata[scene_id] = {
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'lat': lat,
                'lon': lon,
            }
    return metadata


def load_scene_dataset(dataset_path: str, metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Собирает список всех изображений из dataset_path с привязанными метаданными из metadata.
    """
    entries = []
    for scene_id in os.listdir(dataset_path):
        scene_path = os.path.join(dataset_path, scene_id)
        if not os.path.isdir(scene_path):
            continue

        meta = metadata.get(scene_id, {})
        lat = meta.get('lat', 0.0)
        lon = meta.get('lon', 0.0)
        title = meta.get('title', '')
        description = meta.get('description', '')

        # Фильтруем только изображения по расширениям
        for fname in os.listdir(scene_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(scene_path, fname)
                entries.append({
                    'scene_id': scene_id,
                    'title': title,
                    'description': description,
                    'path': file_path,
                    'lat': lat,
                    'lon': lon,
                })
    return entries
