import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        _, (hn, _) = self.lstm(x)      # hn: [num_layers, batch, hidden_size]
        last_hidden = hn[-1]          # 마지막 레이어의 hidden state
        out = self.fc(last_hidden)    # [batch, num_classes]
        return out
    
# class AttentionLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=5):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
#         #self.attn = nn.Linear(hidden_size, 1)
#         self.fc = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Linear(hidden_size, num_classes)
#         )

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)                             # [batch, seq_len, hidden]
#         attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq_len, 1]
#         context = torch.sum(attn_weights * lstm_out, dim=1)       # [batch, hidden]
#         out = self.fc(context)                                 # [batch, num_classes]
#         return out                                              # raw logits
