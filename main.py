#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from app.media_loader import MediaLoader

# 🔧 Защита от OpenMP конфликта между FAISS и PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from app.vpr_system import VPRSystem
from app.config import CONFIG


def test_single_image(path):
    print(f"\n🔍 Тест изображения: {path}")
    vpr = VPRSystem()
    vpr.build_index(CONFIG['database_images'])

    img = MediaLoader.load_image(path)
    if img:
        result = vpr.search(img)
        if result:
            print("\n🎯 Найдено:")
            print(f"  📸 Путь: {result['image_path']}")
            print(f"  📍 Координаты: {result['lat']}, {result['lon']}")
            print(f"  📏 Расстояние: {result['distance']}")
        else:
            print("❌ Ничего не найдено")

if __name__ == '__main__':
    # ✅ Путь можно задать напрямую или взять из конфигурации
    image_path = CONFIG.get('query_image', 'vpr_data/kittty_2.jpg')
    
    test_single_image(image_path)
