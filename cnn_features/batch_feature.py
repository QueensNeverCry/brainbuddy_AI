import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'# TF 경고 log 무시
import cv2
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.cnn_encoder import CNNEncoder
from models.face_crop_yolo import crop_face_batch_chunked


# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset 정의
class VideoFolderDataset(Dataset):
    def __init__(self, dataset_link, T=100):
        self.dataset_link = dataset_link
        self.T = T

    def __len__(self):
        return len(self.dataset_link)

    def __getitem__(self, idx):
        frame_folder, label = self.dataset_link[idx]
        img_files = sorted([
            f for f in os.listdir(frame_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if len(img_files) < self.T:
            raise ValueError(f"Not enough frames in {frame_folder}: {len(img_files)} < {self.T}")

        img_paths = [os.path.join(frame_folder, f) for f in img_files[:self.T]]
        frames = []

        imgs = [cv2.imread(p) for p in img_paths]
        crops = crop_face_batch_chunked(imgs, batch_size=2)

        for crop in crops:
            tensor = transform(crop)
            frames.append(tensor)

        frames_tensor = torch.stack(frames)  # [T, 3, 224, 224]
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

# 배치 단위로 feature 추출
@torch.no_grad()
def extract_batch_features(dataloader, model, device):
    all_features = []
    all_labels = []

    for batch_frames, batch_labels in tqdm(dataloader, desc="🚀 배치 처리 중"):
        # batch_frames: [B, T, 3, 224, 224]
        batch_frames = batch_frames.to(device)  # 전체 배치 GPU로 이동
        batch_features = model(batch_frames)    # [B, T, 1280]
        all_features.extend(batch_features.cpu())  # 각각 [T, 1280]인 텐서들
        all_labels.extend(batch_labels)

    return all_features, all_labels

def save_features_as_pkl(dataset_link, save_path, T=100, batch_size=4, num_workers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEncoder().to(device)
    model.eval()

    dataset = VideoFolderDataset(dataset_link, T)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_skip_invalid  # 예외가 발생한 샘플은 무시
    )

    features, labels = extract_batch_features(dataloader, model, device)

    with open(save_path, "wb") as f:
        pickle.dump({
            "features": features,  # [N x Tensor [T, 1280]]
            "labels": labels       # [N x Tensor]
        }, f)

    print(f"[✅ 저장 완료] {save_path} | 총 샘플: {len(features)}")

# collate_fn: 예외 발생 샘플 무시
def collate_skip_invalid(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return torch.empty(0), torch.empty(0)
    frames, labels = zip(*valid)
    return torch.stack(frames), torch.stack(labels)

# 메인 실행
if __name__ == "__main__":
    
    with open("preprocess2/pickle_labels/train/20_01.pkl", "rb") as f:
        dataset_link = pickle.load(f)

    save_features_as_pkl(
        dataset_link,
        save_path="cnn_features/features/train_20_01_batch.pkl",
        T=100,
        batch_size=4,  # GPU 메모리에 따라 조절
        num_workers=2
    )
