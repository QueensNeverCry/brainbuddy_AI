# late fusion 없이 학습
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score
import torch.nn.functional as F

# ------------------ Dataset ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, num_frames=30):
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.samples = []  # (sorted_paths:list[str], label)
        for folder_path, label in data_list:
            if not os.path.isdir(folder_path): continue
            files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if len(files) < self.num_frames: continue
            files.sort()  # ← 여기서 확정
            # 미리 절대경로로 바꿔두기 (join 비용도 제거)
            paths = [os.path.join(folder_path, f) for f in files[:self.num_frames]]
            self.samples.append((paths, label))

        # OpenCV 내부 스레드 비활성(멀티워커와 충돌/과다 스레딩 방지)
        import cv2; cv2.setNumThreads(0)

        # Normalize 파라미터 캐시
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def __len__(self): return len(self.samples)

    def _load_frame(self, p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)  # Tensor (3,H,W) 가정
            return img
        else:
            img = torch.from_numpy(img).permute(2,0,1).float()/255.0
            return img

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        frames = []
        last_ok = None
        for p in paths:
            t = self._load_frame(p)
            if t is None:
                t = last_ok if last_ok is not None else torch.zeros(3,224,224)
            else:
                last_ok = t
            frames.append(t)

        video = torch.stack(frames, dim=0)  # (T,3,H,W)
        return video, torch.tensor(label, dtype=torch.float32)


# ------------------ Model ------------------

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, dropout2d=0.1, proj_dropout=0.4):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=w)

        self.features = backbone.features                # (B*T, 960, h, w)
        # MobileNetV3-Large 분류기 첫 Linear의 in_features = 960
        self.feat_channels = backbone.classifier[0].in_features  # 960

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))      # (B*T, 960, 2, 2)
        self.drop2d  = nn.Dropout2d(dropout2d)

        # 저랭크 보틀넥: 3840 -> 256 -> 512
        flat_dim = self.feat_channels * 2 * 2            # 960*4 = 3840
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(proj_dropout),
            nn.Linear(256, output_dim), nn.GELU()
        )

    def forward(self, x):  # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)                 # (B*T, 960, h, w)
        x = self.avgpool(x)                  # (B*T, 960, 2, 2)
        x = self.drop2d(x)
        x = x.view(B*T, -1)                  # (B*T, 3840)
        x = self.fc(x)                       # (B*T, 512)
        return x.view(B, T, -1)              # (B, T, 512)

class EngagementModelNoFusion(nn.Module):
    def __init__(self, cnn_feat_dim=512, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
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
            labels = labels.int().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("./log/train5/confusion_matrix", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.savefig(f"./log/train5/confusion_matrix/conf_matrix_epoch_{epoch+1}.png")
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
    acc = accuracy_score(all_labels, all_preds)
    return recall, f1, acc

#===== 검증 확률/라벨 수집 함수 추가, 임계값 탐색 함수 추가
def collect_val_probs_and_labels(model_cnn, model_top, loader, device):
    model_cnn.eval(); model_top.eval()
    probs, labels_all = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            feats = model_cnn(videos)
            logits = model_top(feats)
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y = labels.int().cpu().numpy()
            probs.extend(p.tolist())
            labels_all.extend(y.tolist())
    return np.array(probs), np.array(labels_all)

def safe_prec_rec_f1(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    tn = np.sum((y_true==0) & (y_pred==0))
    acc = (tp+tn) / max(len(y_true), 1)
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return acc, prec, rec, f1

def fbeta(prec, rec, beta=2.0):
    if prec==0 and rec==0: return 0.0
    b2 = beta*beta
    return (1+b2)*prec*rec / (b2*prec + rec)

def pick_thresholds_by_val(probs, labels, beta_for_recall=2.0):
    # 0~1 구간 101개로 스캔(충분히 빠르고 안정적)
    thresholds = np.linspace(0.0, 1.0, 101)
    best_acc, t_acc, acc_pack = -1, 0.5, None
    best_fbeta, t_rec, rec_pack = -1, 0.5, None

    for t in thresholds:
        preds = (probs >= t).astype(np.int32)
        acc, prec, rec, f1 = safe_prec_rec_f1(labels, preds)
        f2 = fbeta(prec, rec, beta_for_recall)

        if acc > best_acc:
            best_acc, t_acc = acc, t
            acc_pack = {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "f2":f2}

        if f2 > best_fbeta:
            best_fbeta, t_rec = f2, t
            rec_pack = {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "f2":f2}

    return t_acc, acc_pack, t_rec, rec_pack




# ------------------ Logging helper (append per epoch) ------------------
def append_log_row(csv_path: str, row: dict):
    """Append a single epoch row to CSV.
    - Creates directory if needed
    - Writes header if file doesn't exist
    - If the epoch already exists in the CSV, it replaces that row (no duplicates)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_df = pd.DataFrame([row])

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            prev = pd.read_csv(csv_path)
            # drop same-epoch rows if any, then append
            if 'epoch' in prev.columns:
                prev['epoch'] = prev['epoch'].astype(int)  # 문자열 정렬 이슈 방지
                prev = prev[prev['epoch'] != int(row['epoch'])]
            prev = pd.concat([prev, row_df], ignore_index=True)
            prev = prev.sort_values('epoch')
            prev.to_csv(csv_path, index=False)
        except Exception as e:
            # fallback to simple append
            row_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    patience = 4
    patience_counter = 0
    num_epochs = 12

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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=4)

    cnn = CNNEncoder().to(device)
    model = EngagementModelNoFusion().to(device)
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=1e-4)
    optimizer = torch.optim.AdamW(
    list(cnn.parameters()) + list(model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,   # 주기: 전체 에폭 수
    eta_min=1e-6        # 최소 학습률
    )


    best_val_loss = float('inf')
    best_model_path = None
    best_model_dir = "./log/train5/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs("./log/train5", exist_ok=True)

    checkpoint_path = "./log/train5/last_checkpoint.pt"
    log_csv_path = "./log/train5/train_log4.csv"


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

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(cnn, model, train_loader, criterion, optimizer, device, accumulation_steps=32)
        val_loss = validate(cnn, model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        recall, f1, acc = evaluate_metrics(cnn, model, val_loader, device)
        # --- (추가) 검증 확률 기반 임계값 탐색: Accuracy용/Recall중심(F2)용 ---
        val_probs, val_labels = collect_val_probs_and_labels(cnn, model, val_loader, device)
        thr_acc, acc_pack, thr_rec, rec_pack = pick_thresholds_by_val(val_probs, val_labels, beta_for_recall=2.0)

        print(f"   ↳ Best-ACC thr={thr_acc:.3f} | acc={acc_pack['acc']:.4f}, rec={acc_pack['rec']:.4f}, "
            f"prec={acc_pack['prec']:.4f}, f1={acc_pack['f1']:.4f}, f2={acc_pack['f2']:.4f}")
        print(f"   ↳ Best-RECALL(F2) thr={thr_rec:.3f} | acc={rec_pack['acc']:.4f}, rec={rec_pack['rec']:.4f}, "
            f"prec={rec_pack['prec']:.4f}, f1={rec_pack['f1']:.4f}, f2={rec_pack['f2']:.4f}")

        # log_history.append({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     "recall": recall,
        #     "f1_score": f1
        # })
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc,
            # --- 추가: 임계값/지표 로그 ---
            "thr_acc": float(thr_acc),
            "thr_acc_acc": float(acc_pack["acc"]),
            "thr_acc_recall": float(acc_pack["rec"]),
            "thr_acc_precision": float(acc_pack["prec"]),
            "thr_acc_f1": float(acc_pack["f1"]),
            "thr_acc_f2": float(acc_pack["f2"]),
            "thr_rec_f2": float(rec_pack["f2"]),
            "thr_rec": float(thr_rec),
            "thr_rec_acc": float(rec_pack["acc"]),
            "thr_rec_recall": float(rec_pack["rec"]),
            "thr_rec_precision": float(rec_pack["prec"]),
            "thr_rec_f1": float(rec_pack["f1"]),
        }
        append_log_row(log_csv_path, row)
        print(f"📝 Appended log for epoch {epoch+1} -> {log_csv_path}")

        evaluate_and_save_confusion_matrix(cnn, model, val_loader, device, epoch)
        scheduler.step()

        # --- Best model 저장 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(best_model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'thr_acc': float(thr_acc),
                'thr_rec': float(thr_rec),
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
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'thr_acc': float(thr_acc),
            'thr_rec': float(thr_rec),
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch+1}")

    # # --- 학습 로그 저장 ---
    # log_df = pd.DataFrame(log_history)
    # os.makedirs("./log/train5", exist_ok=True)
    # log_df.to_csv("./log/train5/train_log4.csv", index=False)
    # print("📄 Training log saved to ./log/train5/train_log4.csv")

    # --- Best 모델 불러오기 ---
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"🔁 Loaded best model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")
    else:
        print("⚠️ No best model was saved during training. Skipping best model loading.")

if __name__ == '__main__':
    main()
