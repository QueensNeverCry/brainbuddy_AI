import os
import pickle
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ------------------ train6.py에서 가져온 Dataset ------------------
class VideoFolderDataset(torch.utils.data.Dataset):
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
        video = torch.stack(frames)

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ 데이터 로드 함수 ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    return all_data

# ------------------ 흐림/밝기 계산 ------------------
def calculate_blur_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    return blur, brightness

# ------------------ CNN Encoder (Feature 추출용) ------------------
class CNNEncoder(torch.nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280 * 4 * 4, output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.mean(dim=0).detach().cpu().numpy()  # 시퀀스 평균 feature

# ------------------ EDA 메인 ------------------
if __name__ == "__main__":
    # 데이터 불러오기
    pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl"
    ]
    data_list = load_data(pkl_files)
    dataset = VideoFolderDataset(data_list, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    # loader = DataLoader(
    #     dataset,
    #     batch_size=2,                  # GPU 여유 되면 2→4→8로 올리기
    #     shuffle=False,
    #     num_workers=4,                 # CPU 코어 수에 맞춰 조정
    #     pin_memory=True,
    #     persistent_workers=True,       # 한 번 띄운 워커 재사용
    #     prefetch_factor=4
    # )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)

    cnn = CNNEncoder().eval()

    records = []
    features = []
    labels = []

    for idx, (video, fusion, label) in enumerate(tqdm(loader, desc="EDA 진행중", unit="sample")):
        folder_path, _ = dataset.data_list[idx]

        # 흐림·밝기 계산 (첫 프레임 기준)
        first_frame_path = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])[0]
        img_path = os.path.join(folder_path, first_frame_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blur, brightness = calculate_blur_brightness(img)

        # fusion feature
        fusion_np = fusion.numpy().flatten()

        # CNN feature 추출
        feat = cnn(video)
        features.append(feat)
        labels.append(int(label.item()))

        records.append({
            "folder": folder_path,
            "label": int(label.item()),
            "blur": blur,
            "brightness": brightness,
            **{f"fusion_{i}": fusion_np[i] for i in range(len(fusion_np))}
        })

    df = pd.DataFrame(records)

    # 📊 흐림/밝기 시각화
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="brightness", y="blur", hue="label", palette="coolwarm")
    plt.title("Brightness vs Blur (by Label)")
    plt.savefig("brightness_vs_blur.png")
    plt.close()

    # 📊 Fusion feature 통계
    fusion_cols = [c for c in df.columns if c.startswith("fusion_")]
    fusion_stats = df.groupby("label")[fusion_cols].mean()
    print("Fusion Feature Mean by Label:\n", fusion_stats)

    # 📊 t-SNE 시각화
    features = np.array(features)
    features = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_feats = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=tsne_feats[:,0], y=tsne_feats[:,1], hue=labels, palette="coolwarm")
    plt.title("t-SNE of CNN Features")
    plt.savefig("tsne_cnn_features.png")
    plt.close()

    print("✅ EDA 완료: brightness_vs_blur.png, tsne_cnn_features.png 저장됨")
