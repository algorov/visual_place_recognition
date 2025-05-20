from typing import Optional
from PIL import Image
import cv2
import numpy as np


class MediaLoader:
    """
    Утилитарный класс для загрузки изображений и кадров из видео.
    """

    @staticmethod
    def load_image(path: str) -> Optional[Image.Image]:
        """
        Загружает изображение с диска и конвертирует его в RGB.

        Args:
            path (str): Путь к изображению.

        Returns:
            Optional[Image.Image]: Объект PIL.Image или None в случае ошибки.
        """
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Не удалось загрузить изображение: {path}, ошибка: {e}")
            return None

    @staticmethod
    def load_frame(frame: np.ndarray) -> Optional[Image.Image]:
        """
        Конвертирует кадр OpenCV (BGR) в формат PIL (RGB).

        Args:
            frame (np.ndarray): Кадр из OpenCV (BGR формат).

        Returns:
            Optional[Image.Image]: Объект PIL.Image или None в случае ошибки.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception as e:
            print(f"⚠️ Не удалось обработать кадр видео, ошибка: {e}")
            return None
