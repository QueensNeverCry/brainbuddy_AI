# focal loss with no early stopping
# ... (기존 import 및 함수 정의는 그대로)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from focal_loss_train import FocalLoss
from models.simple_engagement_model import SimpleEngagementModel
from feature_dataset import CNNFeatureDataset
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
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
    print(f"Using device: {device}")

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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

    model = SimpleEngagementModel().to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    writer = SummaryWriter(log_dir='./runs/engagement_experiment')

    num_epochs = 20
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in loop:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1)

            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).detach()
            preds = (probs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).float().view(-1)

                outputs = model(features).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).detach().cpu()
                all_probs.append(probs)
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        val_preds = (all_probs > best_threshold).astype(int)
        val_f1 = f1_score(all_labels, val_preds)
        val_acc = (val_preds == all_labels).mean()

        # cm = confusion_matrix(all_labels, val_preds)
        # plt.figure(figsize=(6, 5))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Validation Confusion Matrix")
        # plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_2.pth')

    writer.close()
    print("Training complete. Best validation loss:", best_val_loss)

if __name__ == '__main__':
    train()
