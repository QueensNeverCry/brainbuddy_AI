import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
"""
입력: (batch_size, 30, 3, 224, 224)

처리: 각 프레임 → MobileNetV2 → 1280차원 벡터

출력: (batch_size, 30, 1280)
"""
class CNNEncoder(nn.Module):
    def __init__(self, cnn_out_dim=1280):
        super().__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.cnn = mobilenet.features #분류기 제외
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output: (1280, 1, 1)

    def forward(self, x):
        """
        x: (batch_size, 30, 3, 224, 224)
        return: (batch_size, 30, cnn_out_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # (B*T, 3, 224, 224)

        # with torch.no_grad():  # 고정하지 않고 학습
        features = self.cnn(x)               # (B*T, 1280, h, w)
        features = self.avgpool(features)    # feature map 을 avg pooling하여 1x1로 축소 -> (B*T, 1280, 1, 1)

        features = features.view(B, T, -1)   # (B, T, 1280) : 다시 시간 순서로 복원
        return features
    
