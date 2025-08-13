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
from torchvision.transforms import InterpolationMode

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



# =========================
# Model
# =========================
class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        assert backbone == "resnet18", "현재 ckpt는 resnet18 기준"
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        # ResNet 원래 모듈 이름 유지 → ckpt의 cnn.conv1/layer*와 일치
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
        self.out_dim = 512
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class CNN_LSTM(nn.Module):
    def __init__(self, backbone="resnet18", hidden=256, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.cnn = CNNEncoder(backbone=backbone)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,   # 512
            hidden_size=hidden,            # 256
            num_layers=num_layers,         # 1
            dropout=0.0,                   # 1층이면 0
            bidirectional=bidirectional,   # False
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        # ckpt는 classifier.1만 존재(Linear 256→1)
        self.classifier = nn.Sequential(nn.Linear(hidden*d, 1))

    def forward(self, x):  # (B,T,3,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        seq, _ = self.lstm(feats)
        pooled = seq.mean(dim=1)
        return self.classifier(pooled).squeeze(1)

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
        # "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/20_02.pkl",
        # "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/20_04.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/another_data.pkl",
    ]
    best_model_path = "C:/Users/user/Downloads/best_model.pth"
    #best_model_path = "./log/train4/best_model/best_model_epoch_4.pt"  # 필요시 수정
    #best_model_path = "./log/train5/best_model/best_model_epoch_3.pt"
    # 데이터
    test_data_list = load_data(test_pkl_files)
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)

    model = CNN_LSTM(backbone="resnet18").to(device)

    ckpt = torch.load(best_model_path, map_location=device)
    sd = ckpt.get("model", ckpt)   # 'model' 키가 있으면 그걸, 없으면 전체를 state_dict로 간주
    # DataParallel 저장본이면 'module.' 접두사 제거 필요할 수 있음:
    from collections import OrderedDict
    new_sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())


    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("missing:", missing, "unexpected:", unexpected)

    model.eval()

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    with torch.inference_mode():
        for videos, labels in tqdm(test_loader, desc="Test"):  # ✅ (videos, labels)만
            videos = videos.to(device)
            #feats = cnn(videos)
            logits = model(videos)  # ✅ fusion 없음
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.25).astype(np.int32)
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

    # 확률 분포 시각화
    probs_class0 = [p for p, y in zip(all_probs, all_labels) if y == 0]
    probs_class1 = [p for p, y in zip(all_probs, all_labels) if y == 1]

    plt.figure()
    plt.hist(probs_class0, bins=20, alpha=0.6, label="Class 0",
            color="skyblue", edgecolor="black")
    plt.hist(probs_class1, bins=20, alpha=0.6, label="Class 1",
            color="salmon", edgecolor="black")
    plt.xlabel("Predicted probability (class=1)")
    plt.ylabel("Count")
    plt.title("Probability distribution by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_dir = "./log/test"
    os.makedirs(save_dir, exist_ok=True)  # ← 베이스 폴더 보장
    prob_hist_path = os.path.join(save_dir, "prob_histogram_by_class.png")
    plt.savefig(prob_hist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📈 Probability histogram saved: {prob_hist_path}")
    
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
