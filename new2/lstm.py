import torch
import torch.nn as nn

class baseLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dynamic_size, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)

        self.fc_dynamic = nn.Sequential(
            nn.Linear(dynamic_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(hidden_size + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_dyn):
        # x_seq: [B, T, D], x_dyn: [B, D_dyn]
        _, (h_n, _) = self.lstm(x_seq)     # h_n: [1, B, H]
        h_seq = h_n.squeeze(0)             # [B, H]

        h_dyn = self.fc_dynamic(x_dyn)     # [B, 16]

        h = torch.cat([h_seq, h_dyn], dim=1)
        out = self.fc_fusion(h)
        return out
