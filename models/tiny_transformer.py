import torch
import torch.nn as nn
from performer_pytorch import Performer

class PerformerConcentrationModel(nn.Module):
    def __init__(self, input_dim=1280, proj_dim=256, depth=2, heads=4, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, proj_dim)

        self.transformer = Performer(
            dim=proj_dim,
            depth=depth,
            heads=heads,
            causal=False,
            dropout=dropout
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 이진 분류 확률 출력
        )

    def forward(self, x):
        """
        x: Tensor [B, 300, input_dim]
        return: Tensor [B, 1] → 예측 확률 (0~1)
        """
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output_head(x)
