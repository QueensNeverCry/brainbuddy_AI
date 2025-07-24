import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
"""
입력: (batch_size, 300, 3, 224, 224)

처리: 각 프레임 → MobileNetV2 → 1280차원 벡터

출력: (batch_size, 300, 1280)
"""
class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=1280):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)

        # CNN feature extractor (pooling 전까지)
        self.cnn = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)  # output: (1280, 1, 1)

    def forward(self, x):
        """
        x: (batch_size, 300, 3, 224, 224)
        return: (batch_size, 300, cnn_out_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)               # (B*T, 3, 224, 224)

        with torch.no_grad():  # feature 추출만 → freeze
            features = self.cnn(x)               # (B*T, 1280, h, w)
            features = self.pool(features)       # (B*T, 1280, 1, 1)

        features = features.view(B, T, -1)       # (B, T, 1280)
        return features