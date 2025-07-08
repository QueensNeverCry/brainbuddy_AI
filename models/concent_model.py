import torch.nn as nn
import torch

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(0))  # (1, T, 1280)
        return torch.sigmoid(self.fc(out[:, -1, :]))  # (1, 1)
