# import torch.nn as nn
# import torch
# from torch.utils.data import DataLoader
# from models.concent_model import EngagementModel
# import pickle
# from videoframe_dataset import VideoEngagementDataset
# from tqdm import tqdm

# # [(영상경로, 라벨)] 리스트 가져오기
# with open("./preprocessed/dataset_link.pkl", "rb") as f:
#     traindataset_link = pickle.load(f)
# with open("./preprocessed/valdataset_link.pkl", "rb") as f:
#     valdataset_link = pickle.load(f)

# # 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def collate_fn(batch):
#     # batch 내의 None을 제거
#     batch = [item for item in batch if item is not None]
#     return torch.utils.data.dataloader.default_collate(batch)

# # dataset_link는 [('extracted_frames/110006/1100062016', 1), ...] 형태라고 가정
# train_dataset = VideoEngagementDataset(traindataset_link, T=10, device=device)
# val_dataset   = VideoEngagementDataset(valdataset_link,   T=10, device=device)


# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, collate_fn=collate_fn)
# val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn)


# # 모델, 손실 함수, 옵티마이저
# model = EngagementModel().to(device)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # 학습 루프
# num_epochs = 5

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     # tqdm으로 progress bar
#     loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

#     for batch_idx, (features, labels) in loop:
#         features = features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

#     # ----------- Validation -----------
#     model.eval()
#     val_loss = 0.0

#     with torch.no_grad():
#         for features, labels in val_loader:
#             features = features.to(device)
#             labels = labels.to(device)

#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#     avg_val_loss = val_loss / len(val_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.concent_model import EngagementModel
from videoframe_dataset import VideoEngagementDataset
import pickle
from tqdm import tqdm

def collate_fn(batch):
    # batch 내의 None 제거
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    # [(영상경로, 라벨)] 리스트 가져오기
    with open("./preprocessed/dataset_link.pkl", "rb") as f:
        train_link = pickle.load(f)
    with open("./preprocessed/valdataset_link.pkl", "rb") as f:
        val_link = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VideoEngagementDataset(train_link, T=10, device=device)
    val_dataset   = VideoEngagementDataset(val_link, T=10, device=device)

    # ✅ 성능 향상을 위한 DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in loop:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

# ✅ Windows-safe 진입점
if __name__ == '__main__':
    train()
