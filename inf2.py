import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v2
import numpy as np

# ================= 1. CNN Model =====================
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(pretrained=True)
        self.features = base_model.features  # feature extractor 부분만 사용
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 1280-dim 출력

    def forward(self, x):  # x: (B, 3, H, W)
        x = self.features(x)
        x = self.pool(x)  # (B, 1280, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 1280)
        return x

# ================= 2. LSTM Model =====================
# Attention 모듈
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

# Engagement 모델 정의 (BiLSTM + Attention)
class EngagementModel(nn.Module):
    def __init__(self, input_size=1280, proj_size=256, hidden_size=256, output_size=1):
        super().__init__()
        self.proj = nn.Linear(input_size, proj_size)
        self.proj_dropout = nn.Dropout(0.2)

        self.bilstm = nn.LSTM(proj_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.lstm_dropout = nn.Dropout(0.3)

        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):  # x: (B, T, 1280)
        x = self.proj(x)  # (B, T, 256)
        lstm_out, _ = self.bilstm(x)  # (B, T, H*2)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        context = self.attn(lstm_out)  # (B, H*2)
        context = self.norm(context)
        context = self.dropout(context)

        out = self.fc(context)  # (B, 1)
        return out


# ================= Model 불러오기 =====================
def load_model(model_path='best_model.pth', device='cpu'):
    model = EngagementModel()
    full_state = torch.load(model_path, map_location=device)

    print("[INFO] Loaded parameter count:", len(full_state))
    print("[INFO] Example parameter keys:", list(full_state.keys())[:5])

    # 이전처럼 lstm_state_dict만 불러오기 (전체 모델의 state_dict가 들어 있음)
    model.load_state_dict(full_state['lstm_state_dict'])

    model.to(device)
    model.eval()
    return model



# 이미지 전처리 및 추론
def infer(image_paths, model, feature_extractor, device='cpu', threshold=0.65):
    import torchvision.transforms as transforms
    from PIL import Image
    import torch

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frames = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            feature = feature_extractor(tensor)  # (1, 1280)
        frames.append(feature)

    sequence = torch.stack(frames, dim=1)  # (1, T, 1280)
    with torch.no_grad():
        output = model(sequence)  # (1, 1)
        prob = torch.sigmoid(output).item()
        label = 1 if prob > threshold else 0

    return prob, label

# 예시 사용
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = CNNFeatureExtractor().to(device)
    feature_extractor.eval()  # 고정

    model = load_model('best_model.pth', device=device)

     # 테스트할 이미지 경로 -> 저는 제 폴더에 있는 30프레임 가지고 했습니다
    folder_path = r"C:/AIhub_frames/train/02-04-78--1-20-23082700000020-01/segment_0"
    image_sequence = [os.path.join(folder_path, f"{i:04d}.jpg") for i in range(30)]

    prob, label = infer(image_sequence, model, feature_extractor, device=device) #확률값, 이진분류라벨(0,1)

    print(f"📊 Engagement 예측 확률: {prob:.4f}")
    print(f"🔎 이진 분류 결과: {'Engaged (1)' if label == 1 else 'Not Engaged (0)'}")