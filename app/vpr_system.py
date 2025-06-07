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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NetVLAD(NUM_CLUSTERS, FEATURE_DIM).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.index = faiss.IndexFlatL2(NUM_CLUSTERS * FEATURE_DIM)
        self.storage = Storage(CONFIG['redis_host'], CONFIG['redis_port'])
        self.storage.flushdb()

        # соответствие дескриптора сцене
        self.descriptor_to_scene: List[str] = []

    def _process_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = self.model(tensor)
        return desc.cpu().numpy().astype('float32')

    def build_index(self, entries: List[Dict[str, Any]]):
        self.storage.flushdb()
        self.index.reset()
        self.descriptor_to_scene = []

        descriptors = []
        for entry in entries:
            try:
                img = Image.open(entry['path']).convert('RGB')
                desc = self._process_image(img)
                descriptors.append(desc)

                scene_id = entry['scene_id']
                descriptor_id = str(self.storage.incr(f'scene:{scene_id}:counter'))

                self.storage.set(f'scene:{scene_id}:{descriptor_id}:desc', desc.tobytes())
                self.descriptor_to_scene.append(scene_id)

                if not self.storage.exists(f'scene:{scene_id}:lat'):
                    self.storage.set(f'scene:{scene_id}:lat', entry['lat'])
                    self.storage.set(f'scene:{scene_id}:lon', entry['lon'])
                    self.storage.set(f'scene:{scene_id}:title', entry['title'])
                    self.storage.set(f'scene:{scene_id}:description', entry['description'])

            except Exception as e:
                print(f"⚠️ Пропуск: {entry['path']}, ошибка: {e}")

        if descriptors:
            self.index.add(np.vstack(descriptors))

    def search(self, query_img: Image.Image) -> Optional[Dict[str, Any]]:
        query = self._process_image(query_img)

        if self.index.ntotal == 0:
            return None

        distances, ids = self.index.search(query, k=5)

        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1 or idx >= len(self.descriptor_to_scene):
                continue

            scene_id = self.descriptor_to_scene[idx]
            lat = self.storage.get(f'scene:{scene_id}:lat')
            lon = self.storage.get(f'scene:{scene_id}:lon')
            title = self.storage.get(f'scene:{scene_id}:title')
            description = self.storage.get(f'scene:{scene_id}:description')

            if None in (lat, lon):
                print(f"⚠️ Нет координат для scene:{scene_id}")
                continue

            return {
                'scene_id': scene_id,
                'distance': float(dist),
                'lat': float(lat),
                'lon': float(lon),
                'title': title.decode() if isinstance(title, bytes) else title,
                'description': description.decode() if isinstance(description, bytes) else description,
            }

        return None
