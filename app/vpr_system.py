# project_root/app/vpr_system.py

from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import faiss
import torch
from torchvision import transforms
from .model import NetVLAD
from .storage.storage import Storage
from .config import CONFIG, NUM_CLUSTERS, FEATURE_DIM, IMAGE_SIZE
from .geometry import is_valid_match

class VPRSystem:
    """
    Visual Place Recognition System
    - Обрабатывает изображения
    - Извлекает дескрипторы через NetVLAD
    - Ищет ближайшие матчи в FAISS индексе
    - Проверяет геометрическое соответствие
    - Извлекает координаты из хранилища
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем NetVLAD модель
        self.model = NetVLAD(NUM_CLUSTERS, FEATURE_DIM).to(self.device).eval()

        # Трансформации для входных изображений
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # FAISS index (можно заменить на IVFFlat, HNSW и др.)
        self.index = faiss.IndexFlatL2(NUM_CLUSTERS * FEATURE_DIM)

        # Хранилище метаинформации (путь, координаты)
        self.storage = Storage(CONFIG['redis_host'], CONFIG['redis_port'])
        self.storage.flushdb()

    def _process_image(self, image: Image.Image) -> np.ndarray:
        """Преобразует изображение в дескриптор с помощью модели"""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = self.model(tensor)
        return desc.cpu().numpy().astype('float32')

    def build_index(self, image_entries: List[Dict[str, Any]]):
        """Построение FAISS индекса и загрузка метаинформации в Redis"""
        self.storage.flushdb()
        self.index.reset()

        descriptors = []
        for entry in image_entries:
            try:
                img = Image.open(entry['path']).convert('RGB')
                desc = self._process_image(img)
                descriptors.append(desc)

                scene_id = str(self.storage.incr('scene_counter'))
                self.storage.set(f'scene:{scene_id}:descriptor', desc.tobytes())
                self.storage.set(f'scene:{scene_id}:path', entry['path'].encode())
                self.storage.set(f'scene:{scene_id}:lat', str(entry.get('lat', '0')))
                self.storage.set(f'scene:{scene_id}:lon', str(entry.get('lon', '0')))

            except Exception as e:
                print(f"⚠️ Пропущено: {e}")

        if descriptors:
            self.index.add(np.vstack(descriptors))

    def search(self, query_img: Image.Image) -> Optional[Dict[str, Any]]:
        """Ищет ближайшие изображения в индексе и проверяет геометрию"""
        query = self._process_image(query_img)
        if self.index.ntotal == 0:
            return None

        distances, ids = self.index.search(query, k=5)

        for dist, idx in zip(distances[0], ids[0]):
            best_id = idx + 1

            img_path = self.storage.get(f'scene:{best_id}:path')
            lat = self.storage.get(f'scene:{best_id}:lat')
            lon = self.storage.get(f'scene:{best_id}:lon')

            db_img = Image.open(img_path.decode()).convert('RGB')
            if is_valid_match(query_img, db_img):
                return {
                    'scene_id': str(best_id),
                    'distance': float(dist),
                    'image_path': img_path.decode(),
                    'lat': float(lat),
                    'lon': float(lon),
                }

        return None
