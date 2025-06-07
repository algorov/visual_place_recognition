import cv2
import numpy as np
from typing import Union
from PIL import Image


def is_valid_match(query_image: Union[np.ndarray, Image.Image],
                   db_image: Union[np.ndarray, Image.Image]) -> bool:
    """
    Проверяет наличие валидного совпадения между query_image и db_image с помощью ORB и RANSAC-гомографии.

    Args:
        query_image (np.ndarray или PIL.Image): Изображение запроса.
        db_image (np.ndarray или PIL.Image): Изображение из базы.

    Returns:
        bool: True, если найдено достаточное количество совпадающих ключевых точек с хорошей гомографией,
              иначе False.
    """

    # Преобразуем PIL.Image в numpy.ndarray, если необходимо
    if isinstance(query_image, Image.Image):
        query_image = np.array(query_image)
    if isinstance(db_image, Image.Image):
        db_image = np.array(db_image)

    # Конвертация изображений в grayscale (если 3 канала)
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    query_gray = to_gray(query_image)
    db_gray = to_gray(db_image)

    # Инициализация ORB-детектора
    orb = cv2.ORB_create()

    # Поиск ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(db_gray, None)

    # Проверяем наличие дескрипторов
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return False

    # Брутфорс матчинг дескрипторов с crossCheck для повышения качества
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Для гомографии нужно минимум 4 совпадения
    if len(matches) < 4:
        return False

    # Извлекаем координаты совпадающих ключевых точек
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Параметры для RANSAC-гомографии
    homography_threshold = 5.0  # максимально допустимое расстояние для inlier
    min_inlier_ratio = 0.3      # минимальное отношение inliers к общему числу матчей

    # Находим гомографию и маску inliers
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)

    if mask is None or not isinstance(mask, np.ndarray):
        return False

    # Вычисляем долю inliers
    inlier_ratio = np.sum(mask) / len(matches)

    return inlier_ratio > min_inlier_ratio
