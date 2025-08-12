# test_infer.py  (late fusion 없이 학습한 모델용 간단 테스트)
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    f1_score
)

# ------------------ Dataset (학습 코드와 동일 형태) ------------------
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


# ------------------ Utils ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)  # [(folder_path, label), ...]
            all_data.extend(data)
    return all_data


# ------------------ Test only ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/20_02.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/20_04.pkl",
    ]
    best_model_path = "./log/train3/best_model/best_model_epoch_1.pt"  # 필요시 수정

    # 데이터
    test_data_list = load_data(test_pkl_files)
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    # 모델 로드
    cnn = CNNEncoder().to(device)
    model = EngagementModelNoFusion().to(device)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(ckpt['cnn_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])  # ✅ 이제 fc.0 크기 일치(64,128)

    cnn.eval()
    model.eval()

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Test"):  # ✅ (videos, labels)만
            videos = videos.to(device)
            feats = cnn(videos)
            logits = model(feats)  # ✅ fusion 없음
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(np.int32)
            labels = labels.int().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"✅ Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # 혼동행렬 저장
    save_dir = "./log/test"
    os.makedirs(os.path.join(save_dir, "confusion_matrix"), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test")
    out_path = os.path.join(save_dir, "confusion_matrix", "conf_matrix_test.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {out_path}")


if __name__ == "__main__":
    main()
