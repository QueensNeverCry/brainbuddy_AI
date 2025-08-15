import torch
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        self.out_dim = 512
        if backbone == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
            self.out_dim = 512
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(m.features), nn.AdaptiveAvgPool2d((1,1)))
            self.out_dim = 1280
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x):  # (B,3,H,W)
        f = self.encoder(x)
        return f.view(f.size(0), -1)

class CNN_LSTM(nn.Module):
    def __init__(self, backbone: str = "resnet18", hidden: int = 256, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        self.cnn = CNNEncoder(backbone=backbone)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=0.0 if num_layers == 1 else 0.2,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden * d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        seq, _ = self.lstm(feats)
        pooled = seq.mean(dim=1)
        return self.head(pooled).squeeze(1)
