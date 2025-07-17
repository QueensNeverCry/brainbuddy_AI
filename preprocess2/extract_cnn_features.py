import os
import cv2
import torch
from tqdm import tqdm
from models.face_detector import extract_face
from models.feature_extractor import extract_cnn_features
import mediapipe as mp
import pickle

# 저장 형식 : 
# {
#     "features": [Tensor [T, D], Tensor [T, D], ...],  # N개 샘플
#     "labels": [tensor(0.), tensor(1.), ...]           # N개 레이블
# }

def preprocess_dataset(dataset_link, save_dir, T=300, device=None, save_name="features_labels.pkl"):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    os.makedirs(save_dir, exist_ok=True)

    all_features = []
    all_labels = []

    for i, (frame_folder, label) in enumerate(tqdm(dataset_link)):
        try:
            img_files = sorted([
                f for f in os.listdir(frame_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except FileNotFoundError:
            print(f"[SKIP] {frame_folder}: 경로를 찾을 수 없습니다")
            continue

        if len(img_files) < T:
            print(f"[SKIP] {frame_folder}: insufficient frames")
            continue

        img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]

        faces = []
        last_valid_face = None

        for img_path in img_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[ERROR] Failed to load image: {img_path}")
                continue

            face = extract_face(frame, face_mesh)
            if face is None:
                if last_valid_face is not None:
                    face = last_valid_face
                else:
                    print(f"[SKIP] {img_path}: face not detected")
                    break
            else:
                last_valid_face = face

            faces.append(face)

        if len(faces) < T:
            print(f"[SKIP] {frame_folder}: failed to get enough faces. {len(faces)}개.")
            continue

        # CNN feature 추출
        features = extract_cnn_features(faces, device)

        # 누적 저장
        all_features.append(features.cpu())
        all_labels.append(torch.tensor(label, dtype=torch.float32))

    # 모든 feature와 label을 하나의 파일로 저장
    output_path = os.path.join(save_dir, save_name)
    with open(output_path, "wb") as f:
        pickle.dump({
            "features": all_features,  # 리스트 [N개 Tensor]
            "labels": all_labels       # 리스트 [N개 Tensor or float]
        }, f)

    print(f"[✅ Saved] {output_path} | 총 샘플 수: {len(all_features)}")

# -----------------------------
# 실행부
# -----------------------------
with open("./AIHub_label_mapping.pkl", "rb") as f:
    train_link = pickle.load(f)
# with open("../preprocess/val_link.pkl", "rb") as f:
#     val_link = pickle.load(f)

preprocess_dataset(train_link, save_dir="cnn_features/train", T=300, save_name="train_features_labels.pkl")
#preprocess_dataset(val_link, save_dir="cnn_features/valid", T=300, save_name="val_features_labels.pkl")

print(f"총 학습 데이터 수: {len(train_link)}개")
#print(f"총 검증 데이터 수: {len(val_link)}개")

