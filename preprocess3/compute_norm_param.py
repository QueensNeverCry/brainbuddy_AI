# Normalize 파라미터 계산
import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ✅ 사용자 정의 Dataset (이미지 폴더에서 30프레임 이미지 로드)
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = []
        self.transform = transform

        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                self.data_list.append(folder_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path = self.data_list[idx]
        img_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        img_files = img_files[:30]
        images = []

        for fname in img_files:
            img_path = os.path.join(folder_path, fname)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        video = torch.stack(images)  # (30, 3, H, W)
        return video

# ✅ mean/std 계산 함수
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    mean = 0.
    std = 0.
    total_frames = 0

    for videos in tqdm(loader, desc="📊 Calculating mean/std"):
        # videos: (B, 30, 3, H, W)
        B, T, C, H, W = videos.shape
        frames = videos.view(B * T, C, H, W)  # (B*T, 3, H, W)

        frames = frames.view(frames.size(0), frames.size(1), -1)  # (N, C, H*W)
        mean += frames.mean(2).sum(0)  # sum over all pixels per channel
        std += frames.std(2).sum(0)
        total_frames += frames.size(0)

    mean /= total_frames
    std /= total_frames
    return mean, std


# ✅ main
if __name__ == "__main__":
    import pickle

    # 학습 데이터 리스트가 저장된 pkl 파일
    train_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl",
    ]

    all_data = []
    for path in train_pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)

    # transform 정의 (정규화 없음!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = VideoFolderDataset(all_data, transform=transform)
    mean, std = compute_mean_std(dataset)

    print("\n✅ 계산된 평균 (mean):", mean.tolist())
    print("✅ 계산된 표준편차 (std):", std.tolist())
