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

# ------------------ train6.py에서 가져온 Dataset + folder_path 반환 ------------------
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
        video = torch.stack(frames)  # (T, 3, 224, 224)

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        # B안: 폴더 경로까지 함께 반환
        return video, fusion, torch.tensor(label, dtype=torch.float32), folder_path

# ------------------ 데이터 로드 함수 ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    return all_data

# ------------------ 흐림/밝기 계산 ------------------
def calculate_blur_brightness(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    return blur, brightness

# ------------------ CNN Encoder (배치 유지) ------------------
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

    def forward(self, x):  # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)                 # (B*T, D)
        x = x.view(B, T, -1).mean(1)   # (B, D)  ← 시퀀스 평균만
        return x

# ------------------ EDA 메인 ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    loader = DataLoader(
        dataset,
        batch_size=16,                  # 여유 되면 4→8로 점진 확대
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    cnn = CNNEncoder().to(device).eval()
    torch.backends.cudnn.benchmark = True

    records = []
    features = []
    labels = []

    # 샘플 단위 진행률을 위해 total=len(dataset)로 pbar 구성
    with tqdm(total=len(dataset), desc="EDA 진행중", unit="sample") as pbar:
        for videos, fusions, labels_batch, folders in loader:
            # 선택: 프레임 다운샘플 (속도 ↑)
            # videos = videos[:, ::3]

            videos = videos.to(device, non_blocking=True)  # (B, T, 3, 224, 224)

            # EDA는 추론모드 + (GPU면) AMP 권장
            if device.type == "cuda":
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    feats_batch = cnn(videos)  # (B, D)
            else:
                with torch.inference_mode():
                    feats_batch = cnn(videos)

            B = videos.size(0)
            for b in range(B):
                folder_path = folders[b]
                label_int   = int(labels_batch[b].item())
                fusion_np   = fusions[b].numpy().flatten()
                feat_np     = feats_batch[b].detach().cpu().numpy()

                # 흐림/밝기: 첫 프레임 한 장으로 측정
                first_frame = sorted([f for f in os.listdir(folder_path)
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[0]
                img = cv2.imread(os.path.join(folder_path, first_frame))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                blur, brightness = calculate_blur_brightness(img)

                labels.append(label_int)
                features.append(feat_np)
                records.append({
                    "folder": folder_path,
                    "label": label_int,
                    "blur": blur,
                    "brightness": brightness,
                    **{f"fusion_{i}": fusion_np[i] for i in range(len(fusion_np))}
                })

            pbar.update(B)  # 샘플 단위로 진행률 업데이트

    df = pd.DataFrame(records)

    # ====== fusion 분석/시각화 & CSV 저장 ======
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    os.makedirs("eda_outputs", exist_ok=True)

    # 1) 원시 레코드 저장 (샘플 단위)
    df.to_csv("eda_outputs/samples_with_fusion.csv", index=False, encoding="utf-8-sig")

    # 2) fusion 통계 (라벨별): mean/std/max/min + CSV 저장
    fusion_cols = [c for c in df.columns if c.startswith("fusion_")]
    fusion_stats = df.groupby("label")[fusion_cols].agg(["mean", "std", "max", "min"])

    # 멀티인덱스 컬럼 평탄화 후 저장
    fusion_stats_flat = fusion_stats.copy()
    fusion_stats_flat.columns = [f"{col}_{stat}" for col, stat in fusion_stats.columns]
    fusion_stats_flat.to_csv("eda_outputs/fusion_stats_by_label.csv", encoding="utf-8-sig")

    print("Fusion Feature Stats by Label:\n", fusion_stats)

    # 3) fusion 차원 간 상관관계 히트맵 (전체)
    plt.figure(figsize=(6, 5))
    corr = df[fusion_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Fusion Correlation (All Samples)")
    plt.tight_layout()
    plt.savefig("eda_outputs/fusion_corr_all.png", dpi=200)
    plt.close()

    # (선택) 라벨별 상관관계 히트맵
    for lab, sub in df.groupby("label"):
        if len(sub) < 3:  # 표본이 너무 적으면 스킵
            continue
        plt.figure(figsize=(6,5))
        sns.heatmap(sub[fusion_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Fusion Correlation (Label={lab})")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/fusion_corr_label_{lab}.png", dpi=200)
        plt.close()

    # 4) 라벨별 fusion 분포(바이올린/박스)
    for c in fusion_cols:
        plt.figure(figsize=(7,5))
        sns.violinplot(data=df, x="label", y=c, inner="box", cut=0)
        plt.title(f"{c} Distribution by Label")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{c}_violin_by_label.png", dpi=200)
        plt.close()

    # 5) 라벨별 평균±표준편차 (에러바)
    fusion_means = df.groupby("label")[fusion_cols].mean().reset_index()
    fusion_stds  = df.groupby("label")[fusion_cols].std().reset_index()

    # 막대 + 에러바(각 fusion 차원별로 한 그림)
    for c in fusion_cols:
        plt.figure(figsize=(7,5))
        means = fusion_means[["label", c]].set_index("label")[c]
        errs  = fusion_stds[["label", c]].set_index("label")[c]
        means.plot(kind="bar", yerr=errs, capsize=4, alpha=0.8)
        plt.ylabel(c)
        plt.title(f"{c} Mean ± STD by Label")
        plt.tight_layout()
        plt.savefig(f"eda_outputs/{c}_mean_std_by_label.png", dpi=200)
        plt.close()

    # 6) PCA로 fusion 벡터 2D 투영 (색: 라벨)
    X = df[fusion_cols].values
    y = df["label"].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_std)

    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="coolwarm", alpha=0.35)
    plt.title(f"Fusion PCA (2D)  • VarExp= PC1 {pca.explained_variance_ratio_[0]:.2f}, PC2 {pca.explained_variance_ratio_[1]:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("eda_outputs/fusion_pca_2d.png", dpi=200)
    plt.close()

    #7) (옵션) t-SNE로 fusion 벡터 투영 (시간 오래 걸리면 샘플링)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=500)
    X_tsne = tsne.fit_transform(X_std)
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette="coolwarm", alpha=0.35)
    plt.title("Fusion t-SNE (2D)")
    plt.tight_layout()
    plt.savefig("eda_outputs/fusion_tsne_2d.png", dpi=200)
    plt.close()

    print("✅ Fusion 분석 결과 저장 완료:",
        "\n- eda_outputs/samples_with_fusion.csv",
        "\n- eda_outputs/fusion_stats_by_label.csv",
        "\n- eda_outputs/fusion_corr_all.png (및 라벨별 파일)",
        "\n- eda_outputs/<fusion_i>_violin_by_label.png",
        "\n- eda_outputs/<fusion_i>_mean_std_by_label.png",
        "\n- eda_outputs/fusion_pca_2d.png")

    # 📊 흐림/밝기 시각화 (라벨별 색상)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="brightness", y="blur", hue="label", palette="coolwarm",alpha=0.3)
    plt.title("Brightness vs Blur (by Label)")
    plt.savefig("brightness_vs_blur.png")
    plt.close()

    # 📊 Fusion feature 통계
    fusion_cols = [c for c in df.columns if c.startswith("fusion_")]
    fusion_stats = df.groupby("label")[fusion_cols].agg(["mean", "std", "max", "min"])
    print("Fusion Feature Stats by Label:\n", fusion_stats)

    # 📊 t-SNE 시각화 (빠르게 하려면 차원축소나 샘플링 고려)
    features_arr = np.array(features)
    features_arr = StandardScaler().fit_transform(features_arr)
    tsne = TSNE(n_components=2, random_state=42, max_iter=500, perplexity=20)  # 기본보다 가볍게
    tsne_feats = tsne.fit_transform(features_arr)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_feats[:, 0], y=tsne_feats[:, 1], hue=labels, palette="coolwarm",alpha=0.3)
    plt.title("t-SNE of CNN Features")
    plt.savefig("tsne_cnn_features.png")
    plt.close()

    print("✅ EDA 완료: brightness_vs_blur.png, tsne_cnn_features.png 저장됨")
