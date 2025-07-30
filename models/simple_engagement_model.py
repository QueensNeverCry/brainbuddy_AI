import torch.nn as nn
import torch.nn.init as init

class SimpleEngagementModel(nn.Module):
    def __init__(self, input_size=1280, proj_size=256, hidden_size=128, output_size=1):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.proj = nn.Linear(input_size, proj_size)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(proj_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)

        self.fc = nn.Linear(hidden_size, output_size)

        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):  # x: (B, T, 1280)
        x = self.norm(x)
        x = self.proj(x)                      # (B, T, 256)
        x = self.dropout1(x)

        lstm_out, _ = self.lstm(x)            # (B, T, H)
        x = self.dropout2(lstm_out[:, -1, :]) # 마지막 time step 만 사용 (B, H)

        out = self.fc(x)                      # (B, 1)
        return out