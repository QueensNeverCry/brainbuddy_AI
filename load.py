# performance_diagnosis.py

import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

# 1) Dataset 클래스 & 라벨 파싱
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
        return video, torch.tensor(label, dtype=torch.int64)

def parse_label_from_name(folder_name: str) -> int:
    parts = folder_name.split('_')
    return 1 if len(parts) >= 8 and parts[7] == 'F' else 0

# 2) 검증 DataLoader 생성
def make_val_loader(batch_size=16, num_workers=4):
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

    # 동일한 split (shuffle+0.8)
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

# 3) 모델 로드
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("best_checkpoint.pth", map_location=device)

cnn = CNNEncoder().to(device)
cnn.load_state_dict(ckpt['cnn'], strict=False)
cnn.eval()

model = EngagementModel().to(device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# 4) 평가
def evaluate():
    val_loader = make_val_loader()
    all_labels = []
    all_preds  = []

    LOGIT_THRESHOLD = -0.219
    PROB_THRESHOLD = 1 / (1 + np.exp(-LOGIT_THRESHOLD))

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            feats  = cnn(videos)
            logits = model(feats).cpu().numpy().flatten()
            probs  = 1 / (1 + np.exp(-logits))
            preds  = (probs >= PROB_THRESHOLD).astype(int)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())

    # 혼동 행렬 & 리포트
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate()
