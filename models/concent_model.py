# BiLSTM + attention 모델
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)  # Bidirectional → hidden*2

    def forward(self, lstm_out):
        # lstm_out: (1, T, hidden*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (1, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (1, hidden*2)
        return context  # (1, hidden*2)

class EngagementModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (1, T, input_size)
        lstm_out, _ = self.bilstm(x)  # (1, T, hidden*2)
        context = self.attn(lstm_out)  # (1, hidden*2)
        out = torch.sigmoid(self.fc(context))  # (1, 1)
        return out






# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size=1280, hidden_size=256, output_size=1):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.lstm(x.unsqueeze(0))  # (1, T, 1280), 그래서 out의 형태 : (1,T,256)
#         # 실제 train에서는 수정 필요 batch가 현재느 1
#         return torch.sigmoid(self.fc(out[:, -1, :]))  # (1, 1)



