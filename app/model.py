import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch import Tensor


class NetVLAD(nn.Module):
    def __init__(self, num_clusters: int, feature_dim: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim

        # Кластеризация: свёртка для soft-assignment
        self.conv = nn.Conv2d(self.feature_dim, self.num_clusters, kernel_size=1, bias=False)

        # Центроиды кластеров — обучаемые параметры
        self.centroids = nn.Parameter(torch.randn(self.num_clusters, self.feature_dim))

        # Backbone ResNet50 без последних слоёв (pool + fc)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: тензор размера (B, 3, H, W)
        :return: VLAD вектор (B, num_clusters * feature_dim)
        """
        features = self.backbone(x)  # (B, C, H', W')
        B, C, H, W = features.shape

        # Получаем soft assignment (B, num_clusters, H', W')
        soft_assign = F.softmax(self.conv(features), dim=1)

        # Преобразуем к виду (B, C, H'*W')
        x_flat = features.view(B, C, -1)  # (B, C, N)
        soft_assign = soft_assign.view(B, self.num_clusters, -1)  # (B, K, N)

        # Вычисляем VLAD: для каждого кластера k считаем сумму по пространству
        vlad = []
        for k in range(self.num_clusters):
            residual = x_flat - self.centroids[k].view(1, C, 1)  # (B, C, N)
            weighted_residual = soft_assign[:, k].unsqueeze(1) * residual  # (B, C, N)
            vlad_k = weighted_residual.sum(dim=2)  # (B, C)
            vlad.append(vlad_k)

        vlad = torch.stack(vlad, dim=1)  # (B, K, C)
        vlad = vlad.view(B, -1)  # (B, K*C)

        # L2 нормализация по фичам
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
