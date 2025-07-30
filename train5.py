import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

# 1) Dataset 정의
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: List of (folder_path, label)
        """
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
        if len(imgs) < 30:
            raise ValueError(f"'{folder_path}'에 이미지 30장 미만")
        imgs = imgs[:30]

        frames = []
        for fn in imgs:
            img = Image.open(os.path.join(folder_path, fn)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video = torch.stack(frames)  # (30, 3, H, W)
        return video, torch.tensor(label, dtype=torch.float32)

# 2) 라벨 파싱 (폴더명에서 F=1, 그 외=0)
def parse_label_from_name(folder_name: str) -> int:
    parts = folder_name.split('_')
    if len(parts) < 8:
        raise ValueError(f"Invalid folder name: '{folder_name}'")
    return 1 if parts[7] == 'F' else 0

# 3) Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # logits, targets: (B,1)
        bce = self.bce(logits, targets)
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

# 4) Train/Eval 함수
def train_epoch(loader, cnn, model, criterion, optimizer):
    cnn.train()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for videos, labels in tqdm(loader, desc="Train"):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B,1)

        optimizer.zero_grad()
        feats = cnn(videos)
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate_epoch(loader, cnn, model, criterion):
    cnn.eval()
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Valid"):
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)  # (B,1)
            feats = cnn(videos)
            logits = model(feats)
            loss = criterion(logits, labels)

            # flatten and convert to Python scalars
            batch_logits = logits.detach().cpu().numpy().flatten().tolist()
            batch_labels = labels.cpu().numpy().flatten().tolist()
            all_logits.extend(batch_logits)
            all_labels.extend(batch_labels)

            total_loss += loss.item()
            total_samples += labels.size(0)

    # 종료 후 메트릭 계산
    avg_loss = total_loss / len(loader)
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    roc_auc = roc_auc_score(all_labels, probs) if len(set(all_labels)) > 1 else float('nan')
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(all_labels, preds)
    report = classification_report(all_labels, preds, digits=4)
    sample_preds = list(zip(all_logits, probs.tolist(), preds.tolist(), all_labels))[:5]
    accuracy = (preds == np.array(all_labels)).mean()

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'report': report,
        'sample_preds': sample_preds
    }

# 5) Main
def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 디렉토리 설정
    base_dirs = [
        r"C:\Users\user\Downloads\126.eye\01-1.data\Training\01.data\TS\001\T1\image_30_face_crop",
        r"C:\Users\user\Downloads\126.eye\01-1.data\Training\01.data\TS\002\T1\image_30_face_crop",
        r"C:\Users\user\Downloads\126.eye\01-1.data\Training\01.data\TS\003\T1\image_30_face_crop",
    ]

    full_list = []
    for base in base_dirs:
        for d in os.listdir(base):
            folder_path = os.path.join(base, d)
            if os.path.isdir(folder_path):
                label = parse_label_from_name(d)
                full_list.append((folder_path, label))

    random.shuffle(full_list)
    n_train = int(0.8 * len(full_list))
    train_list, val_list = full_list[:n_train], full_list[n_train:]

    # Transform 정의
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    train_ds = VideoFolderDataset(train_list, transform=train_transform)
    val_ds   = VideoFolderDataset(val_list,   transform=val_transform)

    # Oversampling sampler
    train_labels = [lbl for _, lbl in train_ds.data_list]
    counts = np.bincount(train_labels)
    class_weights = 1. / counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=4)

    # 모델 초기화
    from models.cnn_encoder import CNNEncoder
    from models.engagement_model import EngagementModel
    cnn   = CNNEncoder().to(device)
    model = EngagementModel().to(device)

    # Loss, Optimizer, Scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    best_auc = 0.0
    early_stop_patience = 5
    patience_counter = 0

    # 학습 루프
    for epoch in range(1, 21):
        print(f"\n=== Epoch {epoch} ===")
        tr_loss, tr_acc = train_epoch(train_loader, cnn, model, criterion, optimizer)
        print(f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")

        val_stats = validate_epoch(val_loader, cnn, model, criterion)
        print(f"Val   Loss: {val_stats['loss']:.4f}, Acc: {val_stats['accuracy']:.4f}, ROC AUC: {val_stats['roc_auc']:.4f}")
        print("\n📊 Confusion Matrix:\n", val_stats['confusion_matrix'])
        print("\n📋 Classification Report:\n", val_stats['report'])
        print("\n🔍 Sample preds:")
        for logit, prob, pred, true in val_stats['sample_preds']:
            print(f"logit={logit:.3f}, prob={prob:.3f}, pred={pred}, true={true}")

        scheduler.step(val_stats['loss'])

        # Early stopping 및 체크포인트 저장
        if val_stats['roc_auc'] > best_auc:
            best_auc = val_stats['roc_auc']
            patience_counter = 0
            torch.save({
                'cnn': cnn.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "best_checkpoint.pth")
            print(">> 모델 성능 향상, 체크포인트 저장")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f">> {early_stop_patience} epochs 개선 없음, 학습 종료")
                break

if __name__ == "__main__":
    main()
