import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import timm
from torchvision import transforms
from tqdm import tqdm

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# EfficientNetV2-L 모델 불러오기
model = timm.create_model('efficientnetv2_l', pretrained=True, num_classes=0)
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Mediapipe 설정
mp_holistic = mp.solutions.holistic

# 랜드마크 추출 함수
def extract_landmarks(image, holistic_model):
    results = holistic_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    def to_np(landmark_list):
        if landmark_list:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]).flatten()
        else:
            return np.zeros(0)
    face = to_np(results.face_landmarks)
    pose = to_np(results.pose_landmarks)
    left_hand = to_np(results.left_hand_landmarks)
    right_hand = to_np(results.right_hand_landmarks)
    return np.concatenate([
        face if face.size else np.zeros(468*3),
        pose if pose.size else np.zeros(33*3),
        left_hand if left_hand.size else np.zeros(21*3),
        right_hand if right_hand.size else np.zeros(21*3)
    ])

# ROI crop 함수
def crop_region(image, landmarks, indices, expand_ratio=0.2):
    if len(indices) == 0:
        return np.zeros((224,224,3), dtype=np.uint8)
    h, w, _ = image.shape
    coords = landmarks.reshape(-1,3)[indices, :2]
    coords[:,0] *= w
    coords[:,1] *= h
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0)
    width = x2 - x1
    height = y2 - y1
    x1 = max(0, int(x1 - width * expand_ratio))
    y1 = max(0, int(y1 - height * expand_ratio))
    x2 = min(w, int(x2 + width * expand_ratio))
    y2 = min(h, int(y2 + height * expand_ratio))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((224,224,3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (224,224))
    return crop

# CNN 특징 추출 함수
def extract_cnn_features(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy().squeeze()

# 행동 특징 임시 추출 함수
def extract_behavior_features(landmarks):
    return np.random.rand(15)

# Mediapipe 인덱스\FACE_IDX = list(range(0,468))
POSE_IDX = list(range(468, 468+33))
LEFT_HAND_IDX = list(range(468+33, 468+33+21))
RIGHT_HAND_IDX = list(range(468+33+21, 468+33+21+21))

# 전체 처리 함수
def process_and_save_feature(image_path, holistic_model, data_root, save_root):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로딩 실패: {image_path}")
        return
    landmarks = extract_landmarks(image, holistic_model)
    behavior_feat = extract_behavior_features(landmarks)
    face_crop = crop_region(image, landmarks, FACE_IDX)
    left_hand_crop = crop_region(image, landmarks, LEFT_HAND_IDX)
    right_hand_crop = crop_region(image, landmarks, RIGHT_HAND_IDX)
    pose_crop = crop_region(image, landmarks, POSE_IDX)

    face_feat = extract_cnn_features(face_crop)
    left_hand_feat = extract_cnn_features(left_hand_crop)
    right_hand_feat = extract_cnn_features(right_hand_crop)
    pose_feat = extract_cnn_features(pose_crop)

    full_feature = np.concatenate([
        landmarks,
        behavior_feat,
        face_feat,
        left_hand_feat,
        right_hand_feat,
        pose_feat
    ])

    relative_path = os.path.relpath(image_path, data_root)
    save_path = os.path.join(save_root, relative_path.replace(".jpg", ".npy"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, full_feature)

# 이미지 경로들을 처리
def process_all_images(image_root, save_root):
    holistic = mp_holistic.Holistic(static_image_mode=True)
    for root, _, files in os.walk(image_root):
        for file in tqdm(files):
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                process_and_save_feature(image_path, holistic, image_root, save_root)
    holistic.close()

if __name__ == "__main__":
    image_root = "C:/KSEB/brainbuddy_AI/frames/train"
    save_root = "C:/KSEB/brainbuddy_AI/models/features"
    process_all_images(image_root, save_root)
