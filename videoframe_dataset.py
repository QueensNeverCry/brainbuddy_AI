from torch.utils.data import Dataset
from detection.face_detector import detect_faces, extract_face
from models.feature_extractor import extract_cnn_features
import cv2
import torch
import os
import glob
import mediapipe as mp

class VideoEngagementDataset(Dataset):
    def __init__(self, dataset_link, T=10, device=None):
        self.dataset_link = dataset_link
        self.T = T
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1, #감지할 최대 얼굴 개수
            refine_landmarks=True,  # 눈, 입, 동공 등 정밀도 향상(468개 랜드마크 추적)
            min_detection_confidence=0.4,  # default:0.5
            min_tracking_confidence=0.4
        )

    def __len__(self):
        return len(self.dataset_link)

    def __getitem__(self, idx):
        frame_folder, label = self.dataset_link[idx]

        # 경로가 존재하는지 체크
        if not os.path.exists(frame_folder):
            print(f"{frame_folder} 경로가 존재하지 않습니다. 건너뜁니다.")
            return None  # 경로가 없으면 건너뛰기
        
        img_files = sorted([
            f for f in os.listdir(frame_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(img_files) < self.T:
            raise ValueError(f"{frame_folder} 내 프레임 이미지 수가 {self.T}장보다 적습니다.")
        
        img_paths = [os.path.join(frame_folder, f) for f in img_files[:self.T]]
        
        faces = []
        failed_files = []  # 문제 발생한 파일들 기록
        last_valid_face = None
        for img_path in img_paths:
            frame = cv2.imread(img_path)
            
            # 이미지 로드 확인
            if frame is None:
                raise ValueError(f"{img_path} 이미지 로드 실패")
                continue
            
            face = extract_face(frame, self.face_mesh)
            
            if face is None:
                # 얼굴을 감지하지 못한 경우, 이전에 감지된 얼굴을 사용
                if last_valid_face is not None:
                    face = last_valid_face
                    print(f"{img_path} 에서 얼굴 감지 실패. 이전 얼굴 사용.")
                else:
                    # 처음부터 얼굴을 감지할 수 없는 경우는 오류 발생
                    cv2.imwrite("debug_failed_frame.jpg", frame)
                    print(f"{img_path} 에서 얼굴 감지 실패. debug_failed_frame.jpg 저장됨")
                    failed_files.append(img_path)
                    continue
            else:
                # 얼굴을 감지한 경우, 마지막으로 감지된 얼굴을 저장
                last_valid_face = face
            
            faces.append(face)

        if len(faces) < self.T:  # 얼굴을 모두 감지하지 못했으면 예외 발생
            raise ValueError(f"{frame_folder}에서 {self.T}개의 얼굴을 감지하지 못했습니다. 실패한 파일들: {failed_files}")
    
        feature_sequence = extract_cnn_features(faces, self.device)
        
        return feature_sequence, torch.tensor([label], dtype=torch.float32)
