from collections import deque
from typing import Tuple
import numpy as np


class PositionFilter:
    """
    Скользящий медианный фильтр для координат (широта, долгота).
    Поддерживает окно фиксированного размера для сглаживания данных.
    """
    def __init__(self, window: int = 5):
        self.window = window
        self.lat_buf = deque(maxlen=window)
        self.lon_buf = deque(maxlen=window)

    def update(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Добавляет новую координату и возвращает медианные значения текущего окна.
        """
        self.lat_buf.append(lat)
        self.lon_buf.append(lon)
        median_lat = float(np.median(list(self.lat_buf)))
        median_lon = float(np.median(list(self.lon_buf)))
        return median_lat, median_lon

    def clear(self) -> None:
        """Очищает буферы."""
        self.lat_buf.clear()
        self.lon_buf.clear()

    def set_window(self, window: int) -> None:
        """Меняет размер окна и сбрасывает буферы."""
        self.window = window
        self.lat_buf = deque(maxlen=window)
        self.lon_buf = deque(maxlen=window)
