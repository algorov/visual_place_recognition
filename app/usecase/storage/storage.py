import redis
import json
import numpy as np

from typing import Optional
from ...domain.model import SceneMetadata


# --- Константы шаблонов ключей ---
SCENE_DESCRIPTOR_KEY = lambda scene_id, desc_id: f"{scene_id}:{desc_id}:desc"
SCENE_KEY_TEMPLATE = "scene:{}"


class RedisStub:
    """Заглушка Redis, если сервер недоступен. Работает полностью в памяти."""
    def __init__(self):
        self._data = {}
        self._counters = {}

    def ping(self):
        return True

    def incr(self, key):
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    def set(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def exists(self, key):
        return key in self._data

    def flushdb(self):
        self._data.clear()
        self._counters.clear()


class Storage:
    """
    Обёртка для Redis с fallback на RedisStub.
    Предоставляет высокоуровневые методы для работы с метаданными сцен.
    """
    def __init__(self, host: str, port: int):
        self._client = self._connect(host, port)

    def _connect(self, host: str, port: int):
        try:
            client = redis.Redis(host=host, port=port, decode_responses=True)
            client.ping()
            print("✅ Redis подключен")
            return client
        except redis.ConnectionError:
            print("⚠️ Redis недоступен, используется RedisStub")
            return RedisStub()

    # --- Внутренние методы ---

    def _set(self, key, value):
        self._client.set(key, value)

    def _get(self, key):
        return self._client.get(key)

    def _exists(self, key):
        if isinstance(self._client, redis.Redis):
            return self._client.exists(key) > 0
        return self._client.exists(key)

    # --- Публичный API ---

    def flush(self):
        self._client.flushdb()

    def next_id(self, key: str) -> int:
        return self._client.incr(key)

    def scene_exists(self, scene_id: str) -> bool:
        return self._exists(SCENE_KEY_TEMPLATE.format(scene_id))

    def set_descriptor(self, scene_id: str, desc_id: str, desc: np.ndarray):
        self._set(SCENE_DESCRIPTOR_KEY(scene_id, desc_id), desc.tobytes())

    def set_scene_metadata(self, scene_id: str, metadata: SceneMetadata) -> None:
        key = SCENE_KEY_TEMPLATE.format(scene_id)
        self._set(key, json.dumps(metadata.__dict__))

    def get_scene_metadata(self, scene_id: str) -> Optional[SceneMetadata]:
        key = SCENE_KEY_TEMPLATE.format(scene_id)
        raw = self._get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return SceneMetadata(
                scene_id=scene_id,
                title=data.get("title", ""),
                description=data.get("description", ""),
                latitude=float(data.get("latitude", 0.0)),
                longitude=float(data.get("longitude", 0.0)),
            )
        except Exception as e:
            print(f"❌ Ошибка чтения сцены {scene_id}: {e}")
            return None
