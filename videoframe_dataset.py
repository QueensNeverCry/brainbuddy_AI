from torch.utils.data import Dataset
from detection.face_detector import detect_faces, extract_face
from models.feature_extractor import extract_cnn_features
import cv2
import torch
import os
import glob

class VideoEngagementDataset(Dataset):
    def __init__(self, dataset_link, T=10, device='cpu'):
        self.dataset_link = dataset_link
        self.T = T
        self.device = device

    def __len__(self):
        return len(self.dataset_link)

    def __getitem__(self, idx):
        frame_folder, label = self.dataset_link[idx]
        
        # 해당 폴더의 이미지 파일 리스트를 얻음 (확장자 무관, 정렬 포함)
        img_files = sorted([
            f for f in os.listdir(frame_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(img_files) < self.T:
            raise ValueError(f"{frame_folder} 내 프레임 이미지 수가 {self.T}장보다 적습니다.")
        
        # 첫 T개 이미지 경로 리스트
        img_paths = [os.path.join(frame_folder, f) for f in img_files[:self.T]]
        
        faces = []
        for img_path in img_paths:
            frame = cv2.imread(img_path)
            face = extract_face(frame)  # mediapipe 얼굴 추출 함수 사용
            if face is None:
                raise ValueError(f"{img_path} 에서 얼굴 감지 실패")
            faces.append(face)
        
        # CNN 특징 추출
        feature_sequence = extract_cnn_features(faces, self.device)  # (T, 1280)
        
        return feature_sequence, torch.tensor([label], dtype=torch.float32)
