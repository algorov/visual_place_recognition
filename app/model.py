import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import ResNet50_Weights


class FeatureExtractor(nn.Module):
    """
    Извлекает пространственные признаки из изображения с помощью ResNet50.
    выход: тензор формы (B, feature_dim, H, W)
    """
    def __init__(self, feature_dim: int = 2048):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Убираем слои классификации
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        features = self.model(x)
        # features: (B, feature_dim, H, W)
        return features


class SpatialTransformer(nn.Module):
    """
    Обрабатывает пространственные признаки через TransformerEncoder.
    Принимает фичи формы (B, C, H, W), возвращает (B, C, N).
    """
    def __init__(self, feature_dim: int = 2048, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=feature_dim, nhead=nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C, H, W)
        B, C, H, W = features.shape
        # Плоская последовательность
        flat = features.view(B, C, H * W)  # (B, C, N)
        seq = flat.permute(2, 0, 1)        # (N, B, C)
        # Пропускаем через трансформер
        out = self.transformer(seq)       # (N, B, C)
        # Возвращаем к форме (B, C, N)
        processed = out.permute(1, 2, 0)   # (B, C, N)
        return processed


class NetVLADAggregator(nn.Module):
    """
    Агрегатор NetVLAD: получает (B, C, N), возвращает дескриптор (B, K*C).
    """
    def __init__(self, num_clusters: int = 64, feature_dim: int = 2048):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.conv = nn.Conv1d(feature_dim, num_clusters, kernel_size=1, bias=False)
        self.centroids = nn.Parameter(torch.randn(num_clusters, feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        soft_assign = F.softmax(self.conv(x), dim=1)    # (B, K, N)
        B, K, N = soft_assign.shape
        C = self.feature_dim
        # вычисляем остатки
        x_exp = x.unsqueeze(1)                          # (B, 1, C, N)
        c_exp = self.centroids.unsqueeze(0).unsqueeze(-1)  # (1, K, C, 1)
        residual = x_exp - c_exp                        # (B, K, C, N)
        # взвешивание
        weight = soft_assign.unsqueeze(2)               # (B, K, 1, N)
        vlad = (weight * residual).sum(dim=-1)          # (B, K, C)
        vlad = vlad.view(B, -1)                         # (B, K*C)
        # L2-нормализация
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class CNNTransformerNetVLAD(nn.Module):
    """
    Полная модель: извлечение признаков -> трансформер -> NetVLAD дескриптор
    """
    def __init__(self,
                 num_clusters: int = 64,
                 feature_dim: int = 2048,
                 nhead: int = 8,
                 num_layers: int = 6):
        super().__init__()
        self.extractor = FeatureExtractor(feature_dim)
        self.transformer = SpatialTransformer(feature_dim, nhead, num_layers)
        self.aggregator = NetVLADAggregator(num_clusters, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feats = self.extractor(x)          # (B, C, H, W)
        processed = self.transformer(feats)  # (B, C, N)
        desc = self.aggregator(processed)   # (B, K*C)
        return desc
