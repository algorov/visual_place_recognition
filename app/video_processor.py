import cv2
from PIL import Image

class VideoProcessor:
    def __init__(self, path: str, step: int = 10):
        self.cap = cv2.VideoCapture(path)
        self.step = step

    def frames(self):
        idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % self.step == 0:
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        self.cap.release()
