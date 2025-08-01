import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (hn, _) = self.lstm(x)       # hn: [num_layers, batch, hidden_size]
        out = self.fc(hn[-1])           # 마지막 레이어 hidden state
        return out                      # [batch, num_classes]
