import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

# ✅ (1) Dataset 정의
class FaceSequenceDataset(Dataset):
    def __init__(self, csv_path, transform=None, img_per_seq=30):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.img_per_seq = img_per_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        folder = row['folder']
        label = int(row['predicted_label'])

        images = []
        image_files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_files) < self.img_per_seq:
            raise ValueError(f"📉 이미지 부족: {folder}에는 {len(image_files)}장만 있음")

        for fname in image_files[:self.img_per_seq]:
            img_path = os.path.join(folder, fname)
            with Image.open(img_path) as img:
                if self.transform:
                    img = self.transform(img)
                images.append(img)

        # (30, C, H, W) → 시퀀스 반환
        images_tensor = torch.stack(images)
        return images_tensor, label

# ✅ (2) 이미지 전처리 (CLIP 기준)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# ✅ (3) CSV 경로 설정 (📌 너가 직접 지정해줘야 함!)
csv_path = 'vlm_labeled_results_binary.csv'  # 예: 같은 디렉토리에 있을 경우

# ✅ (4) Dataset & DataLoader 생성
dataset = FaceSequenceDataset(csv_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ✅ (5) 작동 확인
for batch in dataloader:
    x, y = batch  # x: (B, 30, 3, 224, 224), y: (B)
    print("🔹 입력 시퀀스 shape:", x.shape)
    print("🔹 라벨:", y)
    break
