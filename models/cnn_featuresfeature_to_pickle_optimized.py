# feature_to_pickle_optimized.py

import os
import pickle
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from facenet_pytorch import MTCNN

# 1) Dataset 정의
class FrameDataset(Dataset):
    def __init__(self, frame_paths):
        self.paths = frame_paths
        self.loader = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return idx, self.loader(img), path  # index, tensor image, original path

# 2) 환경 세팅
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN: GPU에서 배치 단위 얼굴 crop
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)

# MobileNet 백본 (평가 모드)
backbone = models.mobilenet_v2(pretrained=True).to(device)
backbone.eval()

# Global Average Pooling 함수
def extract_features(x):
    with torch.no_grad():
        feats = backbone.features(x)          # (B, C, H, W)
        pooled = feats.mean([2,3])            # (B, C)
    return pooled

# 3) 프레임 리스트 로딩
frame_dir = '/path/to/frames'
all_frames = sorted([os.path.join(frame_dir,f)
                     for f in os.listdir(frame_dir)
                     if f.lower().endswith('.jpg')])
dataset = FrameDataset(all_frames)

# 4) DataLoader: num_workers로 I/O & 전처리 병렬화
loader = DataLoader(dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=lambda batch: batch)

# 5) 배치 단위 처리
features_dict = {}
for batch in tqdm(loader, desc='Extracting features'):
    idxs, imgs, paths = zip(*batch)
    imgs = torch.stack(imgs).to(device)               # (B,3,224,224)
    # 얼굴 검출 & crop: 반환 텐서 (B,3,224,224) or None
    faces = mtcnn(imgs)                              
    # None 제거
    valid = [(i,p,f) for i,p,f in zip(idxs, paths, faces) if f is not None]
    if not valid:
        continue
    vi, vp, vf = zip(*valid)
    batch_faces = torch.stack(vf).to(device)
    # 임베딩 추출
    embs = extract_features(batch_faces).cpu().numpy()
    for i, path, emb in zip(vi, vp, embs):
        features_dict[path] = emb

# 6) 피클 저장
with open('features.pkl', 'wb') as f:
    pickle.dump(features_dict, f)

print(f"Saved features for {len(features_dict)} frames.")
