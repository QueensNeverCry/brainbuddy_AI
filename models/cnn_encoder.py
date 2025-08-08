import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

"""
입력: (batch_size, 30, 3, 224, 224)

처리: 각 프레임 → MobileNetV2 → 1280차원 벡터

출력: (batch_size, 30, 1280)
"""
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)

        self.cnn = mobilenet.features  # 분류기 제외
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 출력 크기 조절
        self.fc = nn.Linear(1280 * 2 * 2, output_dim)

        # (선택) weights.transforms()로 preprocessing 정의 가능

    def forward(self, x):
        """
        x: (batch_size, 30, 3, 224, 224)
        return: (batch_size, 30, cnn_out_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)            # (B*T, 3, 224, 224)
        features = self.cnn(x)                # (B*T, 1280, H', W')
        features = self.avgpool(features)     # (B*T, 1280, 4, 4)
        features = features.view(B * T, -1)   # (B*T, 20480)
        features = self.fc(features)          # (B*T, output_dim)
        features = features.view(B, T, -1)    # (B, T, output_dim)
        return features
