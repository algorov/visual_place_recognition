import pytest
from app.viewpoint import ViewpointDetector
import cv2
import os

@pytest.fixture
def detector():
    return ViewpointDetector()

def test_viewpoint_detection(detector):
    """Тест определения ракурса на тестовых изображениях"""
    test_dir = "test_data/viewpoints"
    for view in ['front', 'left', 'right']:
        img_path = os.path.join(test_dir, f"{view}.jpg")
        img = cv2.imread(img_path)
        result = detector.detect(img)
        assert result['type'] == view, f"Ошибка для {view}"