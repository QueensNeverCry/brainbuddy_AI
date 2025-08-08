import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

# 1) Dataset 정의
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        imgs = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        imgs = imgs[:30] if len(imgs) >= 30 else imgs + imgs[:30-len(imgs)]

        frames = []
        for fn in imgs:
            img = Image.open(os.path.join(folder_path, fn)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video = torch.stack(frames)  # (30, 3, H, W)
        return video, torch.tensor(label, dtype=torch.float32)

# 2) 라벨 파싱
def parse_label_from_name(folder_name: str) -> int:
    parts = folder_name.split('_')
    return 1 if len(parts) >= 8 and parts[7] == 'F' else 0

# 3) Focal Loss 정의 (이전 최적 alpha=0.75, gamma=1.0 유지)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        b = self.bce(logits, targets)
        p_t = torch.exp(-b)
        loss = self.alpha * (1 - p_t) ** self.gamma * b
        return loss.mean() if self.reduction == "mean" else loss.sum()

# 4) 모델 정의: Dropout & LayerNorm 추가
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, videos):
        # videos: (B, 30, 3, H, W)
        B, T, C, H, W = videos.shape
        x = videos.view(-1, C, H, W)
        f = self.features(x)
        f = self.pool(f).view(B, T, -1)
        f = f.mean(dim=1)       # (B, D)
        f = self.dropout(f)
        return f

class EngagementModel(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = self.dropout(x)
        return self.fc2(x)

# 5) Train/Eval 함수
def train_epoch(loader, cnn, model, criterion, optimizer):
    cnn.train(); model.train()
    total_loss = total_correct = total_samples = 0
    for videos, labels in tqdm(loader, desc="Train"):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        feats  = cnn(videos)
        logits = model(feats)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_loss   += loss.item()
        total_samples += labels.size(0)

    return total_loss/len(loader), total_correct/total_samples

def validate_epoch(loader, cnn, model, criterion, threshold= -0.219):
    cnn.eval(); model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Valid"):
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)

            feats  = cnn(videos)
            logits = model(feats)
            loss   = criterion(logits, labels)

            # 로짓 threshold 적용
            preds = (logits >= threshold).float()
            total_correct += (preds==labels).sum().item()
            total_loss   += loss.item()
            total_samples += labels.size(0)

    return total_loss/len(loader), total_correct/total_samples

# 6) main: Dropout/LayerNorm 포함 학습 스크립트
def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dirs = [
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop"
    ]

    full_list = []
    for base in base_dirs:
        for d in os.listdir(base):
            path = os.path.join(base, d)
            if os.path.isdir(path):
                full_list.append((path, parse_label_from_name(d)))
    random.shuffle(full_list)
    n_train = int(0.8*len(full_list))
    train_list, val_list = full_list[:n_train], full_list[n_train:]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.RandomAffine(0, translate=(0.1,0.1)),
        transforms.ColorJitter(0.2,0.2,0.2), transforms.GaussianBlur(3),
        transforms.ToTensor(), transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = VideoFolderDataset(train_list, transform=train_transform)
    val_ds   = VideoFolderDataset(val_list,   transform=val_transform)

    labels = [lbl for _,lbl in train_ds.data_list]
    weights = 1./np.bincount(labels)
    sampler = WeightedRandomSampler([weights[l] for l in labels], len(labels), True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=4)

    # 모델 초기화
    cnn   = CNNEncoder().to(device)
    model = EngagementModel().to(device)

    criterion = FocalLoss(alpha=0.75, gamma=1.0)
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    

    best_val_loss = float('inf')
    patience=5; no_improve=0
    for epoch in range(1,21):
        print(f"\n=== Epoch {epoch} ===")
        tr_loss, tr_acc = train_epoch(train_loader, cnn, model, criterion, optimizer)
        print(f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")
        val_loss, val_acc = validate_epoch(val_loader, cnn, model, criterion)
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
        if val_loss<best_val_loss:
            best_val_loss=val_loss; no_improve=0
            torch.save({'cnn':cnn.state_dict(),'model':model.state_dict()},"best_reg.pth")
            print(">> 검증 손실 개선, 저장")
        else:
            no_improve+=1; print(f">> {no_improve} epochs no improve")
            if no_improve>=patience: break

if __name__=="__main__": main()
