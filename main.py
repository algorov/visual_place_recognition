#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uvicorn
from app.utils.env_patch import apply_openmp_patch
from app.iface.server import VPEServer
from app.usecase.loader.scene_loader import load_scene_dataset, load_scene_metadata


def load_entries():
    """
    Загружает метаданные сцен и изображения, формирует список записей для индексации.

    :return: List[Dict[str, Any]] — записи с путями к изображениям и метаданными.
    """
    try:
        metadata = load_scene_metadata('data/scenes_metadata.csv')
        print(f"📁 Загружено метаданных сцен: {len(metadata)}")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке метаданных сцен: {e}")
        metadata = {}

    try:
        entries = load_scene_dataset('data/scenes/', metadata)
        print(f"🖼 Загружено изображений: {len(entries)}")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке изображений сцен: {e}")
        entries = []

    return entries


# Применяем патч для корректной работы с OpenMP (например, для PyTorch + faiss)
apply_openmp_patch()

# Инициализируем и запускаем сервер
app = VPEServer(load_entries()).get_app()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
