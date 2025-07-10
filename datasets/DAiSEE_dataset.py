import os
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from detection.face_detector import detect_faces
# class DAiSEEDataset(Dataset):
#     def __init__(self, clips_info, target_frame_num=20, binary_label=True, device='cpu'):
#         """
#         clips_info: 리스트 of tuples (clip_frames_dir, label)
#             - clip_frames_dir: 프레임 이미지들이 저장된 폴더 경로
#             - label: 원본 집중도 라벨 (0~3)
#         target_frame_num: 모델 입력에 사용할 프레임 수 (ex: 20)
#         binary_label: True면 2,3은 집중(1), 0,1은 비집중(0)
#         device: 'cpu' or 'cuda'
#         """
#         self.clips_info = clips_info
#         self.target_frame_num = target_frame_num
#         self.binary_label = binary_label
#         self.device = device

#         # CNN feature 추출에 사용할 mobilenet 모델 준비
#         self.mobilenet = models.mobilenet_v2(pretrained=True).features.eval().to(self.device)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
        
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])

#     def __len__(self):
#         return len(self.clips_info)

#     def _sample_frames_uniformly(self, frame_paths):
#         total_frames = len(frame_paths)
#         if total_frames == 0:
#             raise ValueError("No frames found in clip folder")

#         if total_frames >= self.target_frame_num:
#             indices = [int(i * total_frames / self.target_frame_num) for i in range(self.target_frame_num)]
#         else:
#             indices = list(range(total_frames)) + [total_frames - 1] * (self.target_frame_num - total_frames)

#         sampled_paths = [frame_paths[i] for i in indices]
#         return sampled_paths

#     def _load_image(self, path):
#         img = cv2.imread(path)
#         if img is None:
#             raise FileNotFoundError(f"Image not found: {path}")
#         return img

#     def _extract_cnn_features(self, frames):
#         # frames: list of numpy arrays (BGR images)
#         tensors = torch.stack([self.transform(f) for f in frames]).to(self.device)  # (T, 3, H, W)
#         with torch.no_grad():
#             features = self.mobilenet(tensors)  # (T, 1280, H', W')
#             pooled = self.avgpool(features)  # (T, 1280, 1, 1)
#             pooled = pooled.view(len(frames), -1)  # (T, 1280)
#         return pooled

#     def __getitem__(self, idx):
#         clip_dir, label = self.clips_info[idx]

#         # 프레임 경로 불러오기 (정렬 필수)
#         frame_names = sorted(os.listdir(clip_dir))
#         frame_paths = [os.path.join(clip_dir, f) for f in frame_names]

#         # 균등 샘플링
#         sampled_paths = self._sample_frames_uniformly(frame_paths)

#         # 프레임 로딩
#         frames = [self._load_image(p) for p in sampled_paths]

#         # CNN feature 추출
#         feature_sequence = self._extract_cnn_features(frames)  # (T, 1280)

#         # 라벨 변환 (binary or multiclass)
#         if self.binary_label:
#             label = 1 if label >= 2 else 0
#         else:
#             label = int(label)

#         return feature_sequence.cpu(), torch.tensor(label, dtype=torch.float if self.binary_label else torch.long)

class DAiSEEDataset(Dataset):
    def __init__(self, video_paths, label_dict, max_frames=100):
        self.video_paths = video_paths
        self.label_dict = label_dict
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        features = detect_faces(video_path, self.max_frames)
        video_name = os.path.basename(video_path)
        label = self.label_dict.get(video_name, 0)
        
        # 모델 입력이 torch.Tensor일 경우
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
