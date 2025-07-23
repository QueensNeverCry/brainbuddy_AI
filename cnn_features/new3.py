import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
from models.face_crop import crop_face
import multiprocessing

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CPU 전용: 얼굴 검출 + transform 처리
def preprocess_only(args):
    frame_folder, label, T = args

    if not os.path.exists(frame_folder):
        return None

    img_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if len(img_files) < T:
        return None

    img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]
    frames = []

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        # warm-up (optional)
        _ = face_detector.process(np.zeros((240, 320, 3), dtype=np.uint8))

        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                return None
            face_crop = crop_face(img, face_detector)
            tensor = transform(face_crop)
            frames.append(tensor)

    if len(frames) != T:
        return None

    frames_tensor = torch.stack(frames)  # [100, 3, 224, 224]
    return frames_tensor, torch.tensor(label, dtype=torch.float32)

# CNN 추론은 여기서 수행
def save_features_as_pkl(dataset_link, save_path, device_str="cuda", T=100, batch_size=8, num_workers=4):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1. 전처리 병렬 실행 (mediapipe + transform)
    print(f"🧠 전처리 병렬 실행 ({num_workers} workers)")
    tasks = [(frame_folder, label, T) for frame_folder, label in dataset_link]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(preprocess_only, tasks), total=len(tasks)))

    # 2. 유효한 결과만 필터링
    valid_results = [(frames, label) for (frames, label) in results if frames is not None]
    print(f"✅ 유효 샘플 수: {len(valid_results)}")

    if not valid_results:
        print("❌ 전처리 결과 없음")
        return

    # 3. CNN 모델 로드 (GPU)
    model = CNNEncoder().to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_results), batch_size), desc="🚀 CNN 배치 추론"):
            batch = valid_results[i:i+batch_size]
            batch_frames = torch.stack([frames for frames, _ in batch]).to(device)  # [B, 100, 3, 224, 224]
            batch_labels = [label for _, label in batch]

            outputs = model(batch_frames).cpu()  # [B, 100, 1280]

            all_features.extend([feat for feat in outputs])
            all_labels.extend(batch_labels)

    # 4. 저장
    with open(save_path, "wb") as f:
        pickle.dump({
            "features": all_features,  # List[Tensor[100, 1280]]
            "labels": all_labels       # List[Tensor]
        }, f)

    print(f"[✅ 저장 완료] {save_path}")
    print(f"총 유효 샘플: {len(all_features)}")

# 진입점
if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()

    with open("preprocess2/pickle_labels/train/20_01.pkl", "rb") as f:
        dataset_link = pickle.load(f)

    save_features_as_pkl(
        dataset_link,
        save_path="cnn_features/features/train_20_01.pkl",
        device_str="cuda",
        T=100,
        batch_size=8,         # GPU 메모리에 따라 조절
        num_workers=4         # CPU 병렬 처리 수
    )
