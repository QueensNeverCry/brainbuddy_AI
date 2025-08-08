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
    def __init__(self, data_list, transform=None):
        self.data_list = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(img_files) >= 30:
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:30]
        frames = []
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)  # BGR 형식
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)  # To apply existing torchvision transforms
            frames.append(self.transform(img_pil))
        video = torch.stack(frames)  # (30, 3, 224, 224)

        # fusion_features가 없으면 0으로 대체
        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Model ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 출력을 (1280, 4, 4) → 20480
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),  # 축소
            nn.ReLU(),
            nn.Dropout(0.3),                # 과적합 방지
            nn.Linear(2048, output_dim),   # 최종 출력 dim으로 압축
            nn.ReLU()                      # optional: 다음 LSTM에서 더 좋은 표현
        )

    def forward(self, x):  # x: (B, 30, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

class EngagementModel(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + fusion_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        _, (hn, _) = self.lstm(cnn_feats)
        x = torch.cat([hn.squeeze(0), fusion_feats], dim=1)
        return self.fc(x)

# ------------------ Training ------------------
def train(model_cnn, model_top, loader, criterion, optimizer, device, accumulation_steps=4):
    model_cnn.train()
    model_top.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (videos, fusion, labels) in enumerate(tqdm(loader, desc="Train")):
        videos, fusion, labels = videos.to(device), fusion.to(device), labels.to(device).unsqueeze(1)

        features = model_cnn(videos)
        output = model_top(features, fusion)
        loss = criterion(output, labels)

        loss.backward()  # Gradient 누적
        total_loss += loss.item()

        # 일정 step마다 optimizer 업데이트
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(loader)

# ------------------ Validate ------------------
def validate(model_cnn, model_top, loader, criterion, device):
    model_cnn.eval()
    model_top.eval()
    total_loss = 0

    with torch.no_grad():
        for videos, fusion, labels in tqdm(loader, desc="Validation"):
            videos, fusion, labels = videos.to(device), fusion.to(device), labels.to(device).unsqueeze(1)
            features = model_cnn(videos)
            outputs = model_top(features, fusion)
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

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, fusion, labels in loader:
            videos, fusion = videos.to(device), fusion.to(device)
            features = model_cnn(videos)
            outputs = model_top(features, fusion)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.savefig(f"./log/confusion_matrix/train1/conf_matrix_epoch_{epoch+1}.png")
    plt.close()
    print(f"📊 Confusion matrix saved: conf_matrix_epoch_{epoch+1}.png")

def evaluate_metrics(model_cnn, model_top, loader, device):
    model_cnn.eval()
    model_top.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, fusion, labels in loader:
            videos, fusion = videos.to(device), fusion.to(device)
            features = model_cnn(videos)
            outputs = model_top(features, fusion)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return recall, f1

def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,num_workers=8, pin_memory=True)

    cnn = CNNEncoder().to(device)
    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=1e-4)

    best_val_loss = float('inf')
    best_model_path = "./log/best_model2.pt"
    checkpoint_path = "./log/last_checkpoint2.pt"
    log_history = []

    start_epoch = 0
    patience =3
    patience_counter =0

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

        # 로그 저장
        recall, f1 = evaluate_metrics(cnn, model, val_loader, device)

        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall": recall,
            "f1_score": f1
        })


        # Confusion Matrix 저장
        evaluate_and_save_confusion_matrix(cnn, model, val_loader, device, epoch)

        # best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")
        else :
            patience_counter+=1
            print(f"Early stopping patience {patience_counter}/{patience}")
            if patience_counter >= patience :
                print(f"==== Early stopping Triggered===")
                break

        # 중간 체크포인트 저장
        torch.save({
            'cnn_state_dict': cnn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # train_log 저장
    log_df = pd.DataFrame(log_history)
    log_df.to_csv("./log/train_log2.csv", index=False)
    print("📄 Training log saved to train_log.csv")

    # best 모델 불러오기
    checkpoint = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"🔁 Loaded best model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")


if __name__ == '__main__':
    main()