import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
<<<<<<< HEAD
from pathlib import Path
=======
from models.engagement_model import EngagementModel
>>>>>>> origin/main
from feature_dataset import CNNFeatureDataset
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter  # Uncomment if tensorboard installed
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

<<<<<<< HEAD
# BiLSTM + Attention 모델
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class EngagementModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context = self.attn(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out
=======
>>>>>>> origin/main

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

<<<<<<< HEAD
    # 프로젝트 루트 기준 경로 설정
    base_path = Path(__file__).parent

    # Feature + Label 포함된 피클 파일 자동 탐색
    feature_dir = base_path / "cnn_features" / "features"
    pkl_paths = sorted(feature_dir.glob("*.pkl"))  # 모든 .pkl 파일 로드
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found in {feature_dir}")
    dataset = CNNFeatureDataset([str(p) for p in pkl_paths])

    total_size = len(dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # writer = SummaryWriter(log_dir='./runs/engagement_experiment')

    num_epochs = 20
    best_val_loss = float('inf')
    patience = 3
=======
    train_dataset = CNNFeatureDataset([
        "./cnn_features/features/train_20_01.pkl",
        "./cnn_features/features/train_20_03.pkl",
        "./cnn_features/features/D_train.pkl",
        "./cnn_features/features/eng.pkl"
    ])
    val_dataset = CNNFeatureDataset([
        "./cnn_features/features/valid_20_01.pkl",
        "./cnn_features/features/valid_20_03.pkl",
        "./cnn_features/features/D_val.pkl"
    ])
    
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True,num_workers=2)

    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # 과적합 방지용 : weight decay 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)# 스케쥴러로 lr 조정
    writer = SummaryWriter(log_dir='./runs/engagement_experiment')

    num_epochs = 20
    best_val_loss = float('inf') 
    patience = 6
>>>>>>> origin/main
    patience_counter = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in loop:
            features = features.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().view(-1)

            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        #print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

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
        #print(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.4f}")
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        print(f"[검증 데이터 레이블 분포] {dict(zip(unique_labels, label_counts))}")

<<<<<<< HEAD
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
=======
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss : {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
>>>>>>> origin/main
        plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label="Positive")
        plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label="Negative")
        plt.title("Sigmoid Output Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

<<<<<<< HEAD
        # writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # writer.add_scalar('Loss/validation', avg_val_loss, epoch)
=======
        plt.hist(outputs.detach().cpu().numpy(), bins=100)
        plt.title("Raw Logits Distribution")
        plt.xlabel("Logit Value")
        plt.ylabel("Count")
        plt.show()


        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

>>>>>>> origin/main
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # writer.close()
    print("Training complete. Best validation loss:", best_val_loss)

if __name__ == '__main__':
    train()
