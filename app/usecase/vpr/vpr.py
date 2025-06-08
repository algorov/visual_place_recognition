from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import faiss
import torch
from torchvision import transforms

from app.usecase.mega_loc.model import MegaLoc
from app.usecase.storage.storage import Storage
from ...domain.model import PlaceRecognizeResult, SceneMetadata
from app.config.config import CONFIG, IMAGE_SIZE


class VPRSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MegaLoc().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Вычисляем размерность выходного дескриптора
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
            output_dim = self.model(dummy).shape[1]

        redis_cfg = CONFIG["redis"]
        self.index = faiss.IndexFlatL2(output_dim)
        self.storage = Storage(redis_cfg["host"], redis_cfg["port"])
        self.storage.flush()
        self.descriptor_to_scene: List[str] = []

    def _process_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            desc = self.model(tensor)
        return desc.cpu().numpy().astype("float32")

    def _update_scene_metadata(self, scene_id: str, entry: Dict[str, Any]):
        if not self.storage.scene_exists(scene_id):
            metadata = SceneMetadata(
                scene_id=scene_id,
                title=entry.get("title", ""),
                description=entry.get("description", ""),
                latitude=entry.get("lat", 0.0),
                longitude=entry.get("lon", 0.0),
            )
            self.storage.set_scene_metadata(scene_id, metadata)

    def build_index(self, entries: List[Dict[str, Any]], batch_size: int = 16):
        self.storage.flush()
        self.index.reset()
        self.descriptor_to_scene = []

        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            images = []
            valid_entries = []

            for entry in batch:
                try:
                    img = Image.open(entry["path"]).convert("RGB")
                    tensor = self.transform(img).unsqueeze(0)
                    images.append(tensor)
                    valid_entries.append(entry)
                except Exception as e:
                    print(f"⚠️ Пропуск изображения {entry['path']}: {e}")

            if not images:
                continue

            with torch.no_grad():
                batch_tensor = torch.cat(images).to(self.device)
                descs = self.model(batch_tensor).cpu().numpy().astype("float32")

                self.index.add(descs)

            for desc, entry in zip(descs, valid_entries):
                scene_id = entry["scene_id"]
                desc_id = str(self.storage.next_id(f"{scene_id}:counter"))

                self.storage.set_descriptor(scene_id, desc_id, desc)
                self.descriptor_to_scene.append(scene_id)
                self._update_scene_metadata(scene_id, entry)

        print(f"✅ Индекс построен: {self.index.ntotal} дескрипторов.")

    def search(self, query_img: Image.Image, max_dist: float = 1.5) -> Optional[PlaceRecognizeResult]:
        query = self._process_image(query_img)

        if self.index.ntotal == 0:
            return None

        distances, ids = self.index.search(query, k=5)

        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1 or idx >= len(self.descriptor_to_scene):
                continue
            if dist > max_dist:
                continue

            scene_id = self.descriptor_to_scene[idx]
            metadata = self.storage.get_scene_metadata(scene_id)

            if metadata is None:
                print(f"⚠️ Нет метаданных для scene:{scene_id}")
                continue

            return PlaceRecognizeResult(metadata=metadata, distance=float(dist))

        return None
