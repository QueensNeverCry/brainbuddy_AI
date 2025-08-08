import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class CNNEncoder(nn.Module):
    """
    입력: (batch_size, 30, 3, 224, 224)
    처리: 각 프레임 → MobileNetV2 → 1280차원 벡터
    출력: (batch_size, 30, 1280)
    Dropout(p=0.3) 추가로 과적합 억제
    """
    def __init__(self, cnn_out_dim=1280, dropout_p=0.3):
        super().__init__()
        # 사전학습된 MobileNetV2 백본 로드 (features만 사용)
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.cnn = mobilenet.features        # 분류기 제외하고 feature extractor만
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # feature map -> 1x1로 축소
        self.dropout = nn.Dropout(p=dropout_p)  # dropout 추가

    def forward(self, x):
        """
        x: (B, T, 3, 224, 224)
        return: (B, T, cnn_out_dim)
        """
        B, T, C, H, W = x.shape
        # 프레임을 배치 차원으로 펼치기
        x = x.view(B * T, C, H, W)

        # 특징 추출
        features = self.cnn(x)                # (B*T, 1280, h, w)
        features = self.avgpool(features)     # (B*T, 1280, 1, 1)
        features = features.view(B, T, -1)    # (B, T, 1280)

        # 시퀀스 차원마다 동일하게 dropout 적용
        features = self.dropout(features)
        return features
