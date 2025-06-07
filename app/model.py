import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as tfm
from app.megaloc_model import MegaLocModel

class MegaLocWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MegaLocModel()

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: изображение (B, 3, H, W)
        :return: дескриптор MegaLoc (B, 8448)
        """
        return self.model(x)  # Выдаёт L2-нормализованный глобальный дескриптор
