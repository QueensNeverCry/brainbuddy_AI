import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from feature_dataset import CNNFeatureDataset
from models.concent_model import EngagementModel
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

    # # Dataset
    # train_dataset = VideoEngagementFeatureDataset("./preprocess/preprocessed_features/train_data")
    # val_dataset   = VideoEngagementFeatureDataset("./preprocess/preprocessed_features/val_data")

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,num_workers=2)
    # val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True,num_workers=2)

    dataset = CNNFeatureDataset([
        "./cnn_features/features/train_20_01.pkl",
        "./cnn_features/features/train_20_03.pkl"
    
    ])
    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    # 랜덤 분할
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드
    )
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)

    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)# 스케쥴러로 lr 조정
    writer = SummaryWriter(log_dir='./runs/engagement_experiment')

    num_epochs = 20
    best_val_loss = float('inf') 
    patience = 3
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
        # # 🔹 임계값 튜닝
        # best_threshold = 0.5
        # best_f1 = 0.0
        # # threshold 튜닝 루프 직전
        # print("예측 확률 샘플:", all_probs[:10])
        # print("정답 레이블 샘플:", all_labels[:10])
        # for t in np.arange(0.1, 0.9, 0.05):
        #     preds = (all_probs > t).astype(int)
        #     f1 = f1_score(all_labels, preds)
        #     print(f"[Threshold: {t:.2f}] F1: {f1:.4f}")  # 🔍 F1 변화 확인
        #     if f1 > best_f1:
        #         best_f1 = f1
        #         best_threshold = t
        # val_f1 = best_f1
        # # 기존 val_f1 계산 뒤에 추가
        # cm = confusion_matrix(all_labels, (all_probs > best_threshold).astype(int))

        # plt.figure(figsize=(6,5))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        # plt.xlabel("Predicted Label")
        # plt.ylabel("True Label")
        # plt.title("Confusion Matrix")
        # plt.show()
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss : {avg_train_loss}, Val Loss: {avg_val_loss:.4f}")
        plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label="Positive")
        plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label="Negative")
        plt.title("Sigmoid Output Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.show()


        # TensorBoard 기록
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:  # ✅
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 모델 저장
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    writer.close()
    print("Training complete. Best validation loss:", best_val_loss)

if __name__ == '__main__':
    train()
