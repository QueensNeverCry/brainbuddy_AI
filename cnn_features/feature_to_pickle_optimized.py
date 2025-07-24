# feature_to_pickle_parallel.py

import os
import cv2
import torch
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.cnn_encoder import CNNEncoder
from models.face_crop    import crop_face

# -----------------------------------------------------------------------------
# 1) CLI 인자
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="병렬 DataLoader + GPU 배치 처리로 얼굴 크롭→CNN 특징 벡터 추출"
)
parser.add_argument("-i", "--label_pkl",
                    required=True,
                    help="(link,label) 리스트가 담긴 pickle 파일 경로")
parser.add_argument("-o", "--output_pkl",
                    default="features.pkl",
                    help="저장할 출력 pickle 경로")
parser.add_argument("-T", "--seq_len", type=int, default=100,
                    help="폴더당 사용할 프레임 수 (기본 100)")
parser.add_argument("-b", "--batch_size", type=int, default=8,
                    help="폴더 단위 배치 크기")
parser.add_argument("-j", "--num_workers", type=int, default=4,
                    help="DataLoader num_workers")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2) Transform 정의
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# -----------------------------------------------------------------------------
# 3) Dataset: 하나의 아이템 = 폴더 → (sequence_tensor, label)
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, dataset_link, seq_len, transform):
        """
        dataset_link: [(frame_folder, label), ...]
        seq_len: 한 시퀀스당 사용할 프레임 수
        """
        self.link = dataset_link
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.link)

    def __getitem__(self, idx):
        folder, label = self.link[idx]
        # 1) 존재 확인
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        # 2) 프레임 경로 읽고 T장만 선택
        imgs = sorted(
            [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg",".png"))]
        )[: self.seq_len]

        # 3) 순회하며 crop+transform
        seq = []
        for fn in imgs:
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                raise IOError(f"Cannot read image: {fn}")
            face = crop_face(img)      # 얼굴 검출 (없으면 원본)
            seq.append(self.transform(face))

        # 4) [T, C, H, W] → 반환
        seq_tensor = torch.stack(seq)           # (T,3,224,224)
        return seq_tensor, torch.tensor(label)

# -----------------------------------------------------------------------------
# 4) DataLoader + 모델 세팅
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 로드
model = CNNEncoder().to(device)
model.eval()

# dataset_link 불러오기
with open(args.label_pkl, "rb") as f:
    dataset_link = pickle.load(f)

dataset = SequenceDataset(dataset_link,
                          seq_len=args.seq_len,
                          transform=transform)

loader = DataLoader(dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True)

# -----------------------------------------------------------------------------
# 5) 추출 루프: 배치 단위로 GPU에서 병렬 처리
# -----------------------------------------------------------------------------
all_features = []
all_labels   = []

with torch.no_grad():
    for seq_batch, label_batch in tqdm(loader, desc="Batch Feature Extraction"):
        # seq_batch: [B, T, 3,224,224]
        seq_batch = seq_batch.to(device)            # pin_memory=True 덕분에 빠름
        feats = model(seq_batch)                    # [B, T, emb_dim]
        feats = feats.cpu()                         # 되돌려 담기
        for i in range(feats.size(0)):
            all_features.append(feats[i])           # Tensor [T, emb_dim]
            all_labels.append(label_batch[i])

# -----------------------------------------------------------------------------
# 6) 저장
# -----------------------------------------------------------------------------
with open(args.output_pkl, "wb") as f:
    pickle.dump({
        "features": all_features,   # 리스트 of Tensor [T, emb_dim]
        "labels":   all_labels      # 리스트 of Tensor
    }, f)

print(f"\n✅ 저장 완료 → {args.output_pkl}")
print(f"   총 샘플: {len(all_features)}    (batch_size={args.batch_size}, workers={args.num_workers})")
