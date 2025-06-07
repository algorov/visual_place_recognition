import cv2
from PIL import Image
from typing import Iterator


class VideoProcessor:
    def __init__(self, video_path: str, step: int = 10):
        """
        Инициализация класса для обработки видеопотока.

        :param video_path: путь к видеофайлу
        :param step: шаг пропуска кадров, чтобы брать каждый step-й кадр для обработки,
                     что уменьшает нагрузку и ускоряет обработку
        """
        self.video_path = video_path
        self.step = max(1, step)  # Гарантируем, что шаг не меньше 1

    def frames(self) -> Iterator[Image.Image]:
        """
        Генератор кадров из видео с заданным шагом.

        Класс последовательно читает кадры из видеофайла с помощью OpenCV,
        пропуская кадры согласно шагу. Для каждого выбранного кадра
        происходит конвертация цветового пространства из BGR в RGB,
        после чего кадр преобразуется в объект PIL.Image.

        Использование генератора позволяет эффективно обрабатывать видео
        без загрузки всех кадров в память сразу.

        :return: генератор PIL.Image для каждого выбранного кадра
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл: {self.video_path}")

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % self.step == 0:
                    # OpenCV использует BGR, PIL ожидает RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(rgb_frame)

                frame_id += 1
        finally:
            cap.release()
