import os
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np

# ✅ 1) SlidingWindowDataset 정의
class SlidingWindowDataset(Dataset):
    def __init__(self, csv_file, root_dirs, transform=None, window=10, stride=5):
        self.transform = transform
        self.samples = []  # (folder_path, start_idx, label)

        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            folder_rel = row['folder']
            label = float(row['predicted_label'])
            folder_path = None
            for rd in root_dirs:
                candidate = os.path.join(rd, folder_rel)
                if os.path.isdir(candidate):
                    folder_path = candidate
                    break
            if folder_path is None:
                continue

            image_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(image_files) < window:
                continue

            max_start = len(image_files) - window + 1
            for start in range(0, max_start, stride):
                self.samples.append((folder_path, start, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, start, label = self.samples[idx]
        all_images = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        selected = all_images[start:start+10]
        frames = []
        for fname in selected:
            img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video = torch.stack(frames)  # (window, C, H, W)
        return video, torch.tensor(label, dtype=torch.float32)

# ✅ 2) Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ✅ 3) CNN + LSTM 모델 정의
class CNNLSTMClassifier(nn.Module):
    def __init__(self, cnn_name='resnet18', hidden_size=256, num_layers=1,
                 bidirectional=False, dropout=0.5):
        super().__init__()
        cnn = getattr(models, cnn_name)(pretrained=True)
        self.feature_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()
        self.cnn = cnn

        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=(dropout if num_layers > 1 else 0.0))
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b*t, c, h, w)
        feats = self.cnn(x)
        feats = feats.view(b, t, -1)
        _, (h_n, _) = self.lstm(feats)
        if self.lstm.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]
        logits = self.classifier(h_last)
        return logits

# ✅ 4) Train / Validation 함수
def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    for videos, labels in tqdm(loader, desc='Train'):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(videos)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += labels.size(0)
    return total_loss / len(loader), total_correct / total_samples

def validate_epoch(loader, model, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc='Valid'):
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(videos)
            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)
    return total_loss / len(loader), total_correct / total_samples

# ✅ 5) Main 실행
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 사용자 설정
    csv_path = r'C:/Users/user/Desktop/brainbuddy_AI/vlm_labeled_results_binary.csv'
    root_dirs = [
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop",
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop"
    ]

    # Dataset 생성
    full_ds = SlidingWindowDataset(csv_path, root_dirs,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.481, 0.4578, 0.4082],
                                 [0.2686, 0.2613, 0.2758])
        ]),
        window=10,
        stride=5
    )

    # 데이터 분할
    lengths = [int(0.7 * len(full_ds)), int(0.15 * len(full_ds)), len(full_ds) - int(0.7 * len(full_ds)) - int(0.15 * len(full_ds))]
    train_ds, val_ds, test_ds = random_split(full_ds, lengths, generator=torch.Generator().manual_seed(42))

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    # 모델, 손실, 옵티마이저
    model = CNNLSTMClassifier(cnn_name='resnet18', hidden_size=256).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 학습
    best_loss, patience_counter, patience = float('inf'), 0, 5
    for epoch in range(1, 21):
        print(f'\n📘 Epoch {epoch}')
        tr_loss, tr_acc = train_epoch(train_loader, model, criterion, optimizer, device)
        print(f'✅ Train loss: {tr_loss:.4f}, acc: {tr_acc:.4f}')
        val_loss, val_acc = validate_epoch(val_loader, model, criterion, device)
        print(f'🧪 Valid loss: {val_loss:.4f}, acc: {val_acc:.4f}')
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print('💾 Checkpoint saved')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('⏹️ Early stopping triggered')
                break

    # 테스트 평가
    test_loss, test_acc = validate_epoch(test_loader, model, criterion, device)
    print(f'\n📊 Test loss: {test_loss:.4f}, acc: {test_acc:.4f}')
