# test_infer.py  (간단 테스트 전용)
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

# ------------------ Dataset (train6.py와 동일 형태) ------------------
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
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            frames.append(self.transform(img_pil))
        video = torch.stack(frames)  # (30, 3, 224, 224)

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
        # 인터넷 없이도 동작하도록 weights=None
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU()
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
    best_model_path = "./log/train1/best_model_4.pt"  # 또는 best_model_epoch_*.pt

    # 데이터
    test_data_list = load_data(test_pkl_files)
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)

    # 모델 로드
    cnn = CNNEncoder().to(device)
    model = EngagementModel().to(device)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(ckpt['cnn_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])
    cnn.eval()
    model.eval()

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for videos, fusion, labels in tqdm(test_loader, desc="Test"):
            videos, fusion = videos.to(device), fusion.to(device)
            feats = cnn(videos)
            logits = model(feats, fusion)
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
