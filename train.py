import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from models.concent_model import EngagementModel
import pickle
from videoframe_dataset import VideoEngagementDataset
from tqdm import tqdm

# [(영상경로, 라벨)] 리스트 가져오기
with open("./preprocessed/dataset_link.pkl", "rb") as f:
    dataset_link = pickle.load(f)

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    # batch 내의 None을 제거
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# dataset_link는 [('extracted_frames/110006/1100062016', 1), ...] 형태라고 가정
dataset = VideoEngagementDataset(dataset_link, T=10, device=device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=collate_fn)#멀티 프로세싱 안하기ㅣ..


# 모델, 손실 함수, 옵티마이저
model = EngagementModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # tqdm으로 progress bar
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (features, labels) in loop:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")