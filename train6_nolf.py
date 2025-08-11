# late fusion 없이 학습
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score

# ------------------ Dataset ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, num_frames=30):
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # data_list: [(folder_path, label), ...]
        self.data_list = []
        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(img_files) >= self.num_frames:
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:self.num_frames]

        frames = []
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)  # BGR
            if img is None:
                # 스킵하지 말고, 빈 프레임을 추가하여 길이 유지
                frames.append(torch.zeros(3, 224, 224))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            frames.append(self.transform(img_pil))

        video = torch.stack(frames)  # (T, 3, 224, 224)
        return video, torch.tensor(label, dtype=torch.float32)

# ------------------ Model ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # (1280, 4, 4) → 20480
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU()
        )

    def forward(self, x):  # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)  # (B, T, D)

class EngagementModelNoFusion(nn.Module):
    def __init__(self, cnn_feat_dim=1280, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logit
        )

    def forward(self, cnn_feats):
        # cnn_feats: (B, T, D)
        _, (hn, _) = self.lstm(cnn_feats)   # hn: (1, B, H)
        x = hn.squeeze(0)                   # (B, H)
        return self.fc(x)                   # (B, 1)

# ------------------ Training ------------------
def train(model_cnn, model_top, loader, criterion, optimizer, device, accumulation_steps=4):
    model_cnn.train()
    model_top.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (videos, labels) in enumerate(tqdm(loader, desc="Train")):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        features = model_cnn(videos)
        output = model_top(features)
        loss = criterion(output, labels)

        loss.backward()
        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(loader)

# ------------------ Validate ------------------
def validate(model_cnn, model_top, loader, criterion, device):
    model_cnn.eval()
    model_top.eval()
    total_loss = 0.0

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Validation"):
            videos = videos.to(device)
            labels = labels.to(device).unsqueeze(1)
            features = model_cnn(videos)
            outputs = model_top(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)

def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    return all_data

def evaluate_and_save_confusion_matrix(model_cnn, model_top, loader, device, epoch):
    model_cnn.eval()
    model_top.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            features = model_cnn(videos)
            outputs = model_top(features)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("./log/train3/confusion_matrix", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.savefig(f"./log/train3/confusion_matrix/conf_matrix_epoch_{epoch+1}.png")
    plt.close()
    print(f"📊 Confusion matrix saved: conf_matrix_epoch_{epoch+1}.png")

def evaluate_metrics(model_cnn, model_top, loader, device):
    model_cnn.eval()
    model_top.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            features = model_cnn(videos)
            outputs = model_top(features)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl"
    ]
    val_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_03.pkl"
    ]

    train_data_list = load_data(train_pkl_files)
    val_data_list = load_data(val_pkl_files)

    train_dataset = VideoFolderDataset(train_data_list)
    val_dataset = VideoFolderDataset(val_data_list)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)

    cnn = CNNEncoder().to(device)
    model = EngagementModelNoFusion().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=1e-4)

    best_val_loss = float('inf')
    best_model_path = None
    best_model_dir = "./log/train3/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs("./log/train3", exist_ok=True)

    checkpoint_path = "./log/train3/last_checkpoint.pt"
    log_history = []

    start_epoch = 0
    patience = 3
    patience_counter = 0

    # --- 체크포인트 불러오기 ---
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(ckpt['cnn_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(cnn, model, train_loader, criterion, optimizer, device, accumulation_steps=32)
        val_loss = validate(cnn, model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        recall, f1 = evaluate_metrics(cnn, model, val_loader, device)
        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall": recall,
            "f1_score": f1
        })

        evaluate_and_save_confusion_matrix(cnn, model, val_loader, device, epoch)

        # --- Best model 저장 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(best_model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, best_model_path)
            print(f"✅ Best model saved: {best_model_path} (val_loss={val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early stopping triggered!")
                break

        # --- 체크포인트 저장 ---
        torch.save({
            'cnn_state_dict': cnn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # --- 학습 로그 저장 ---
    log_df = pd.DataFrame(log_history)
    os.makedirs("./log/train3", exist_ok=True)
    log_df.to_csv("./log/train3/train_log3.csv", index=False)
    print("📄 Training log saved to ./log/train3/train_log3.csv")

    # --- Best 모델 불러오기 ---
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"🔁 Loaded best model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")
    else:
        print("⚠️ No best model was saved during training. Skipping best model loading.")

if __name__ == '__main__':
    main()
