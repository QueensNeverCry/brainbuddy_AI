import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.concent_model import EngagementModel
from video_engagement_feature_dataset import VideoEngagementFeatureDataset
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")

    # Dataset
    train_dataset = VideoEngagementFeatureDataset("./preprocess/preprocessed_features/train_data")
    val_dataset   = VideoEngagementFeatureDataset("./preprocess/preprocessed_features/val_data")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True,num_workers=2)

    # 클래스 불균형 처리 (pos_weight 계산)
    num_neg = 496
    num_pos = 5730
    pos_weight_value = (num_neg / num_pos)*5 # ⬅️ 수정됨
    pos_weight = torch.tensor([pos_weight_value], device=device)

    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)# 스케쥴러로 lr 조정
    writer = SummaryWriter(log_dir='./runs/engagement_experiment')

    num_epochs = 20
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    global_step = 0  # for batch logging

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in loop:
            features = features.to(device,non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().view(-1)

            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 🔹 배치 단위 TensorBoard 기록
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().view(-1)

                outputs = model(features).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).detach().cpu()
                all_probs.append(probs)
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        # 🔍 추가: label 분포 확인
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        print(f"[검증 데이터 레이블 분포] {dict(zip(unique_labels, label_counts))}")
        # 🔹 임계값 튜닝
        best_threshold = 0.5
        best_f1 = 0.0
        # threshold 튜닝 루프 직전
        print("예측 확률 샘플:", all_probs[:10])
        print("정답 레이블 샘플:", all_labels[:10])
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_labels, preds)
            print(f"[Threshold: {t:.2f}] F1: {f1:.4f}")  # 🔍 F1 변화 확인
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        val_f1 = best_f1
        # 기존 val_f1 계산 뒤에 추가
        cm = confusion_matrix(all_labels, (all_probs > best_threshold).astype(int))

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        print(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Best Threshold: {best_threshold:.2f}")

        # TensorBoard 기록
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('F1/validation', val_f1, epoch)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 🔸 모델 저장
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    writer.close()
    print("Training complete. Best F1:", best_val_f1)

if __name__ == '__main__':
    train()
