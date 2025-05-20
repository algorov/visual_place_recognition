import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class NetVLAD(nn.Module):
    def __init__(self, num_clusters, feature_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = feature_dim
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=1, bias=False)
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        features = self.backbone(x)
        B, C, _, _ = features.shape
        soft_assign = torch.softmax(self.conv(features), dim=1)
        x_flat = features.view(B, C, -1)
        soft_assign = soft_assign.view(B, self.num_clusters, -1)

        vlad = torch.stack([
            (soft_assign[:, k].unsqueeze(1) * (x_flat - self.centroids[k].view(1, C, 1))).sum(dim=2)
            for k in range(self.num_clusters)
        ], dim=1)

        vlad = nn.functional.normalize(vlad.view(B, -1), p=2, dim=1)
        return vlad