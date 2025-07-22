import os
import cv2
import torch
import pickle
from tqdm import tqdm
import mediapipe as mp
from torchvision import transforms
from multiprocessing import Pool, cpu_count

from models.cnn_encoder import CNNEncoder
from models.face_crop import crop_face

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
mp_face_detection = mp.solutions.face_detection

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def face_crop_worker(args):
    frame_folder, T = args
    if not os.path.exists(frame_folder):
        return None

    img_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(img_files) < T:
        return None

    img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]
    cropped_tensors = []

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                return None
            face = crop_face(img, face_detector)
            tensor = transform(face)
            cropped_tensors.append(tensor)

    return cropped_tensors

@torch.no_grad()
def save_features_as_pkl_parallel_crop(dataset_link, save_path, device=None, T=100, num_workers=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 8) # (CPU수-4)개 만큼 병렬로

    print(f"🌐 병렬 얼굴 검출 시작 (workers={num_workers})")
    with Pool(processes=num_workers) as pool:
        args = [(frame_folder, T) for frame_folder, _ in dataset_link]
        cropped_results = list(tqdm(pool.imap(face_crop_worker, args), total=len(dataset_link)))

    model = CNNEncoder().to(device)
    model.eval()

    all_features = []
    all_labels = []

    for (cropped, (_, label)) in zip(cropped_results, dataset_link):
        if cropped is None or len(cropped) != T:
            continue
        frames_tensor = torch.stack(cropped).unsqueeze(0).to(device)  # [1, T, 3, 224, 224]
        features = model(frames_tensor).squeeze(0).cpu()  # [T, 1280]
        all_features.append(features)
        all_labels.append(torch.tensor(label, dtype=torch.float32))

    with open(save_path, "wb") as f:
        pickle.dump({
            "features": all_features,
            "labels": all_labels
        }, f)

    print(f"[✅ 저장 완료] {save_path} | 총 샘플: {len(all_features)}")

if __name__ == "__main__":
    with open("preprocess2/pickle_labels/train/20_01.pkl", "rb") as f:
        dataset_link = pickle.load(f)

    save_features_as_pkl_parallel_crop(
        dataset_link,
        save_path="cnn_features/features/train_20_01.pkl",
        T=100
    )
