# ensemble_two_models.py

import os
import random
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    accuracy_score, f1_score
)

# 1) Dataset 정의
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        imgs = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        )
        # 최소 30장 확보
        imgs = imgs[:30] if len(imgs) >= 30 else imgs + imgs[:30-len(imgs)]
        frames = []
        for fn in imgs:
            img = Image.open(os.path.join(folder_path, fn)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        video = torch.stack(frames)  # (30,3,H,W)
        return video, torch.tensor(label, dtype=torch.float32)

# 2) 라벨 파싱
def parse_label_from_name(folder_name: str) -> int:
    parts = folder_name.split('_')
    return 1 if len(parts) >= 8 and parts[7] == 'F' else 0

# 3) Validation DataLoader 준비
def make_val_loader(batch_size=8, num_workers=4):
    base_dirs = [
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
        r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop"
    ]
    full_list = []
    for base in base_dirs:
        for d in os.listdir(base):
            path = os.path.join(base, d)
            if os.path.isdir(path):
                full_list.append((path, parse_label_from_name(d)))

    random.seed(42)
    random.shuffle(full_list)
    n_train = int(0.8 * len(full_list))
    val_list = full_list[n_train:]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_ds = VideoFolderDataset(val_list, transform=val_transform)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 4) 모델 클래스 import
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel

# 5) 체크포인트에서 모델 로드 (strict=False)
def load_cnn_eng_models(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device)

        # CNNEncoder 로드
        cnn = CNNEncoder().to(device)
        raw_cnn = ckpt.get('cnn', ckpt)
        cnn.load_state_dict(raw_cnn, strict=False)
        cnn.eval()

        # EngagementModel 로드
        model = EngagementModel().to(device)
        raw_model = ckpt.get('model', ckpt)
        model.load_state_dict(raw_model, strict=False)
        model.eval()

        models.append((cnn, model))
    return models

# 6) 단일 모델 AUC 계산
def compute_auc_for_model(cnn, model, val_loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            feats  = cnn(videos)
            logits = model(feats).cpu().numpy().flatten()
            probs  = 1 / (1 + np.exp(-logits))
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return roc_auc_score(all_labels, all_probs)

# 7) 앙상블 예측 (Weighted Voting)
def weighted_ensemble_logits(models, videos, weights, device):
    logits_list = []
    with torch.no_grad():
        videos = videos.to(device)
        for (cnn, model), w in zip(models, weights):
            feats  = cnn(videos)
            logits = model(feats).cpu().numpy().flatten()
            logits_list.append(logits * w)
    return np.sum(np.stack(logits_list, axis=0), axis=0)

# 8) 메인
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = make_val_loader()

    # 사용할 체크포인트 두 개
    ckpt_paths = ["best_checkpoint.pth", "best_reg.pth"]
    models = load_cnn_eng_models(ckpt_paths, device)

    # 1) 각 모델 AUC 측정 → 가중치 정규화
    aucs = []
    for (cnn, model), p in zip(models, ckpt_paths):
        auc = compute_auc_for_model(cnn, model, val_loader, device)
        print(f"{p} → Val AUC: {auc:.4f}")
        aucs.append(auc)
    weights = np.array(aucs) / np.sum(aucs)
    print("Normalized weights:", weights.tolist())

    # 2) 앙상블 로짓 수집
    all_logits, all_labels = [], []
    for videos, labels in val_loader:
        logits = weighted_ensemble_logits(models, videos, weights, device)
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.numpy().tolist())

    # 3) 최적 cut-off 재계산
    fpr, tpr, th = roc_curve(all_labels, all_logits)
    best_th = th[np.argmax(tpr - fpr)]
    print(f"▶ Ensemble best logit cutoff: {best_th:.3f}")

    # 4) 최종 지표
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    preds = (probs >= 1/(1+np.exp(-best_th))).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, probs)
    print(f"Ensemble Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
