import os
import cv2
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from models.cnn_encoder import CNNEncoder

# ─── 경고 메시지 숨기기 ─────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─── MediaPipe 얼굴 검출기 초기화 (한 번만) ─────────────────────────
mp_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ─── 이미지 전처리(transform) ───────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─── 통계 카운터 ───────────────────────────────────────────────────
skip_path_count = 0
skip_frame_count = 0
load_fail_count = 0

# ─── DataLoader용 None 제거 함수(모듈 최상단에 정의) ─────────────────
def collate_remove_none(batch):
    """None인 항목을 걸러낸 후 (frames, label) 페어만 남김"""
    return [item for item in batch if item is not None]

# ─── 얼굴 크롭 함수 ─────────────────────────────────────────────────
def crop_face(img_bgr, fallback_to_full=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    results = mp_detector.process(img_rgb)
    if results.detections:
        best = max(
            results.detections,
            key=lambda d: (
                d.location_data.relative_bounding_box.width *
                d.location_data.relative_bounding_box.height
            )
        )
        bbox = best.location_data.relative_bounding_box
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        x2 = min(x1 + int(bbox.width * w), w)
        y2 = min(y1 + int(bbox.height * h), h)
        return img_rgb[y1:y2, x1:x2]
    return img_rgb if fallback_to_full else None

# ─── Dataset 정의 ───────────────────────────────────────────────────
class FrameFolderDataset(Dataset):
    def __init__(self, dataset_link, T=100):
        self.dataset_link = dataset_link
        self.T = T

    def __len__(self):
        return len(self.dataset_link)

    def __getitem__(self, idx):
        global skip_path_count, skip_frame_count, load_fail_count
        folder, label = self.dataset_link[idx]
        if not os.path.exists(folder):
            skip_path_count += 1
            return None
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if len(files) < self.T:
            skip_frame_count += 1
            return None

        tensors = []
        for fname in files[:self.T]:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                load_fail_count += 1
                return None
            face = crop_face(img)
            tensors.append(transform(face))
        return torch.stack(tensors), torch.tensor(label, dtype=torch.float32)

# ─── 메인 기능 함수 ─────────────────────────────────────────────────
def save_features_as_pkl(
    dataset_link,
    save_path,
    device=None,
    T=100,
    batch_size=4,
    num_workers=4
):
    global skip_path_count, skip_frame_count, load_fail_count

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = CNNEncoder().to(device)
    if device.type == 'cuda':
        model.half()
    model.eval()

    dataset = FrameFolderDataset(dataset_link, T)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_remove_none   # 여기서 람다 대신 최상단 함수 사용
    )

    all_features = []
    all_labels = []

    for batch in tqdm(loader, desc="📦 Feature 추출 중"):
        frames, labels = zip(*batch)
        frames = torch.stack(frames)
        if device.type == 'cuda':
            frames = frames.half()
        frames = frames.to(device, non_blocking=True)

        with torch.no_grad():
            feats = model(frames)
        feats = feats.cpu()

        all_features.extend(feats)
        all_labels.extend(labels)

    with open(save_path, "wb") as f:
        pickle.dump({"features": all_features, "labels": all_labels}, f)

    print(f"[✅ 저장 완료] {save_path} | 총 샘플: {len(all_features)}")
    print("\n📊 처리 통계:")
    print(f"  [경로 없음] {skip_path_count}")
    print(f"  [프레임 부족] {skip_frame_count}")
    print(f"  [이미지 로드 실패] {load_fail_count}")
    print(f"  [정상 추출 완료] {len(all_features)}")

# ─── 스크립트 직접 실행 시 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    pkl_path  = sys.argv[1] if len(sys.argv) > 1 else "preprocess2/pickle_labels/valid/20_01.pkl"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "cnn_features/features/valid_20_01.pkl"

    with open(pkl_path, "rb") as f:
        dataset_link = pickle.load(f)
    save_features_as_pkl(dataset_link, save_path=save_path)
