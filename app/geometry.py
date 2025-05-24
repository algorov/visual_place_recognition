import cv2
import numpy as np
from typing import Union
from PIL import Image


def is_valid_match(query_image: Union[np.ndarray, Image.Image], db_image: Union[np.ndarray, Image.Image]) -> bool:
    """
    Проверяет, есть ли валидное совпадение между query_image и db_image с помощью ORB и гомографии.

    Args:
        query_image: Изображение запроса (PIL.Image или numpy.ndarray).
        db_image: Изображение базы (PIL.Image или numpy.ndarray).

    Returns:
        bool: True, если достаточно хорошее совпадение, иначе False.
    """
    # Преобразуем PIL в numpy, если нужно
    if not isinstance(query_image, np.ndarray):
        query_image = np.array(query_image)
    if not isinstance(db_image, np.ndarray):
        db_image = np.array(db_image)

    # Конвертация в grayscale
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY) if query_image.ndim == 3 else query_image
    db_gray = cv2.cvtColor(db_image, cv2.COLOR_RGB2GRAY) if db_image.ndim == 3 else db_image

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(db_gray, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography_threshold = 5.0
    min_inlier_ratio = 0.3

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)

    if mask is None or not isinstance(mask, np.ndarray):
        return False

    inlier_ratio = mask.sum() / len(matches)
    return inlier_ratio > min_inlier_ratio
