from collections import deque
import numpy as np

class PositionFilter:
    def __init__(self, window=5):
        self.lat_buf = deque(maxlen=window)
        self.lon_buf = deque(maxlen=window)

    def update(self, lat, lon):
        self.lat_buf.append(lat)
        self.lon_buf.append(lon)
        return np.median(self.lat_buf), np.median(self.lon_buf)