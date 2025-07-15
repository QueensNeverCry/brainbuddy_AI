# BiLSTM + attention 모델
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)  # Bidirectional → hidden*2

    def forward(self, lstm_out):  # (B, T, H*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)       # (B, H*2)
        return context


class EngagementModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):  # x: (B, T, input_size)
        lstm_out, _ = self.bilstm(x)          # (B, T, H*2)
        context = self.attn(lstm_out)         # (B, H*2)
        context = self.norm(context)          
        context = self.dropout(context)       
        out = self.fc(context) # (B, 1)
        return out



