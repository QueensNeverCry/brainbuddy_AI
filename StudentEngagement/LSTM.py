import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from feature_dataset import CNNFeatureDataset
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class EngagementModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, output_size=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context = self.attn(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out

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
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 데이터셋 경로, 지금까지 만든 피클 파일 위치로 수정
    feature_files = [
    os.path.normpath(r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features.pkl"),
    os.path.normpath(r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features_aug.pkl"),
]

    

    dataset = CNNFeatureDataset(feature_files)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

    model = EngagementModel(input_size=512).to(device)  # 피처 벡터 차원 512에 맞춤
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    writer = SummaryWriter(log_dir=os.path.normpath(r'C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/runs/engagement_experiment'))

    num_epochs = 20
    best_val_loss = float('inf') 
    patience = 3
    patience_counter = 0
    overfit_counter = 0
    global_step = 0
    prev_val_f1 = 0.0
    prev_train_loss = float('inf')

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
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

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

        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        print(f"[검증 데이터 레이블 분포] {dict(zip(unique_labels, label_counts))}")

        # Threshold 0.5 고정으로 F1 계산
        val_preds = (all_probs > 0.5).astype(int)
        val_f1 = f1_score(all_labels, val_preds)
        cm = confusion_matrix(all_labels, val_preds)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.show()

        plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label="Positive")
        plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label="Negative")
        plt.title("Sigmoid Output Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('F1/validation', val_f1, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.normpath(r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Overfitting detection
        if avg_train_loss < prev_train_loss and val_f1 < prev_val_f1:
            overfit_counter += 1
            print(f"⚠️ 과적합 경고: {overfit_counter}회 연속")
        else:
            overfit_counter = 0

        if overfit_counter >= 3:
            print("과적합 판단으로 조기 종료합니다.")
            break

        prev_val_f1 = val_f1
        prev_train_loss = avg_train_loss

    writer.close()
    print("Training complete. Best validation loss:", best_val_loss)

if __name__ == '__main__':
    train()