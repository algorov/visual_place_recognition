import cv2
from PIL import Image
from typing import Iterator


class VideoProcessor:
    def __init__(self, video_path: str, step: int = 10):
        """
        :param video_path: путь к видеофайлу
        :param step: шаг пропуска кадров (берем каждый step-й кадр)
        """
        self.video_path = video_path
        self.step = step

    def frames(self) -> Iterator[Image.Image]:
        """Генератор кадров из видео с шагом self.step."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {self.video_path}")

        try:
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % self.step == 0:
                    # Конвертируем BGR (OpenCV) в RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(rgb_frame)
                frame_id += 1
        finally:
            cap.release()
