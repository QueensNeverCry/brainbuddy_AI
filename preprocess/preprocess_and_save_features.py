import os
import cv2
import torch
from tqdm import tqdm
from models.face_detector import extract_face
from models.feature_extractor import extract_cnn_features
import mediapipe as mp
import pickle
def preprocess_dataset(dataset_link, save_dir, T=10, device=None):
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

    for i, (frame_folder, label) in enumerate(tqdm(dataset_link)):
        feature_save_path = os.path.join(save_dir, f"sample_{i}.pt")
        if os.path.exists(feature_save_path):
            continue  # 이미 전처리된 경우 건너뜀
        try:
            img_files = sorted([
                f for f in os.listdir(frame_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except FileNotFoundError:
            print(f"[SKIP] {frame_folder}: 경로를 찾을 수 없습니다")
            continue

        if len(img_files) < T:#frame이 10개보다 적은 경우 건너뜀
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
        try:
            torch.save({
                'features': features.cpu(),
                'label': torch.tensor([label], dtype=torch.float32)
            }, feature_save_path)
        except Exception as e:
            print(f"[ERROR] Saving failed: {e}")

# 실행 : mediapipe에서 추출한 landmark-> CNN 특징벡터로 미리 추출
# LSTM 훈련 시 사용, 병목 최소화

# git Bash 에서 실행시 PYTHONPATH 설정 후 실행
#   export PYTHONPATH=..
#   python preprocess_and_save_features.py

with open("./train_link.pkl", "rb") as f:
    train_link = pickle.load(f)
with open("./val_link.pkl", "rb") as f:
    val_link = pickle.load(f)

preprocess_dataset(train_link, save_dir="preprocessed_features/train_data", T=10)
preprocess_dataset(val_link, save_dir="preprocessed_features/val_data", T=10)

print(f"총 학습 데이터 수: {len(train_link)}개")
print(f"총 검증 데이터 수: {len(val_link)}개")
