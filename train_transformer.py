import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from models.tiny_transformer import PerformerConcentrationModel

# pkl 불러오기
with open("cnn_features/train_features_labels.pkl", "rb") as f:
    data = pickle.load(f)
    features = data["features"]  # List of [300, 1280] Tensor
    labels = data["labels"]      # List of [1] Tensor (0 or 1)

# Dataset 정의
class BinaryFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.X = [f.float() for f in features]
        self.y = [l.unsqueeze(0).float() for l in labels]  # [1] 형태로

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = BinaryFeatureDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델, 손실, 최적화기
model = PerformerConcentrationModel(input_dim=1280)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        preds = model(batch_x)  # [B, 1] 확률값
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 정확도 계산
        predicted = (preds > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total * 100
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")
