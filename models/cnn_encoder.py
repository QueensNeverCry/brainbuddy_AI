import torch.nn as nn
<<<<<<< HEAD
from torchvision import models

=======
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
"""
입력: (batch_size, 30, 3, 224, 224)
>>>>>>> origin/main

class CNNEncoder(nn.Module):
<<<<<<< HEAD
    def __init__(self, output_dim=512, dropout2d=0.1, proj_dropout=0.4):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=w)

        self.features = backbone.features                # (B*T, 960, h, w)
        self.feat_channels = backbone.classifier[0].in_features  # 960

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))      # (B*T, 960, 2, 2)
        self.drop2d  = nn.Dropout2d(dropout2d)

        flat_dim = self.feat_channels * 2 * 2            # 960*4 = 3840
        
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(proj_dropout),
            nn.Linear(256, output_dim), nn.GELU()
        )

    def forward(self, x):  # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)                 # (B*T, 960, h, w)
        x = self.avgpool(x)                  # (B*T, 960, 2, 2)
        x = self.drop2d(x)
        x = x.view(B*T, -1)                  # (B*T, 3840)
        x = self.fc(x)                       # (B*T, 512)
        return x.view(B, T, -1)              # (B, T, 512)
=======
    def __init__(self, cnn_out_dim=1280):
        super().__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.cnn = mobilenet.features #분류기 제외
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # output: (1280, 1, 1)

    def forward(self, x):
        """
        x: (B, T, 3, 224, 224)
        return: (B, T, cnn_out_dim)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # (B*T, 3, 224, 224)

        # with torch.no_grad():  # 고정하지 않고 학습
        features = self.cnn(x)               # (B*T, 1280, h, w)
        features = self.avgpool(features)    # feature map 을 avg pooling하여 1x1로 축소 -> (B*T, 1280, 1, 1)

        features = features.view(B, T, -1)   # (B, T, 1280) : 다시 시간 순서로 복원

        return features

>>>>>>> origin/main
