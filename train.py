import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from models.concent_model import EngagementModel
import pickle
from videoframe_dataset import VideoEngagementDataset

# [(영상경로, 라벨)] 리스트 가져오기
with open("./preprocessed/dataset_link.pkl", "rb") as f:
    dataset_link = pickle.load(f)

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset_link는 [('extracted_frames/110006/1100062016', 1), ...] 형태라고 가정
dataset = VideoEngagementDataset(dataset_link, T=10, device=device)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


# 모델, 손실 함수, 옵티마이저
model = EngagementModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)      # (batch_size, T, feature_dim)
        labels = labels.to(device)          # (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(features)           # (batch_size, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")