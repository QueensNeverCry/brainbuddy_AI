# BiLSTM + attention 모델
import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_out):  # (B, T, H*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)       # (B, H*2)
        return context


class EngagementModel(nn.Module):
    def __init__(self, input_size=1280,proj_size=256, hidden_size=256, output_size=1):
        super().__init__()
        self.proj = nn.Linear(input_size, proj_size)
        self.proj_dropout = nn.Dropout(0.2)

        self.bilstm = nn.LSTM(proj_size, hidden_size, batch_first=True, bidirectional=True) #True : BiLSTM
        self.lstm_norm = nn.LayerNorm(hidden_size*2)#*2
        self.lstm_dropout = nn.Dropout(0.3)

        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size), # 입력 레이어* 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):  # x: (B, T, 1280)
        x = self.proj(x)    # (B, T, 256)
        lstm_out, _ = self.bilstm(x)          # (B, T, H*2)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        context = self.attn(lstm_out)         # (B, H*2)
        context = self.norm(context)          
        context = self.dropout(context)
        
        #context = lstm_out[:, -1, :]  

        out = self.fc(context) # (B, 1)
        return out

