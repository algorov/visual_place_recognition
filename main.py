#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Any

from app.video_processor import VideoProcessor
from app.position_filter import PositionFilter
from app.scene_loader import load_scene_metadata, load_scene_dataset
from app.vpr_system import VPRSystem
from app.config import CONFIG

# 🔧 Защита от OpenMP конфликта между FAISS и PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


def load_entries() -> List[Dict[str, Any]]:
    """
    Загружает метаданные сцен и формирует список записей с изображениями.

    :return: Список записей с данными сцен.
    """
    try:
        metadata = load_scene_metadata('data/scenes_metadata.csv')
        print(f"📁 Загружено метаданных сцен: {len(metadata)}")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке метаданных сцен: {e}")
        metadata = {}

    entries = load_scene_dataset('data/scenes/', metadata)
    print(f"🖼 Загружено изображений: {len(entries)}")
    return entries


def run_video_inference(entries: List[Dict[str, Any]]):
    """
    Запускает обработку видео, выполняет поиск совпадений по кадрам.

    :param entries: Список записей базы изображений.
    """
    print("\n🎥 Обработка видеофайла:", CONFIG["test_video_path"])

    vpr = VPRSystem()
    vpr.build_index(entries)

    processor = VideoProcessor(CONFIG['test_video_path'], step=10)
    pos_filter = PositionFilter(window=5)

    for frame in processor.frames():
        result = vpr.search(frame)
        if result:
            print("\n🎯 Найдено совпадение:")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")


if __name__ == '__main__':
    entries = load_entries()
    if entries:
        run_video_inference(entries)
    else:
        print("❌ Нет данных для обработки.")

    # Для тестирования одиночного изображения можно раскомментировать:
    # test_single_image("vpr_data/kitty.jpg")
