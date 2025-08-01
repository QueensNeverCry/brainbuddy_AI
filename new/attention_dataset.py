import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AttentionDataset(Dataset):
    def __init__(self, csv_path, seq_len=30):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len

        # ✅ 명시적으로 사용할 feature 리스트 (총 22개)
        self.feature_cols = [
            'head_yaw', 'head_pitch', 'head_roll',
            'cam_distance',
            'gaze_x', 'gaze_y', 'gaze_z',
            'l_eye_x', 'l_eye_y',
            'r_eye_x', 'r_eye_y',
            'l_EAR', 'r_EAR',
            'gaze_stability', 'head_stability', 'gaze_jitter',
            'fixation_duration', 'saccade_frequency',
            'blink', 'distance_change', 'head_tilt_angle'
        ]

        self.sequences = []
        self.labels = []

        # ✅ folder_name 기준 그룹화 → 시퀀스 생성
        grouped = self.df.groupby("folder_name")

        for name, group in grouped:
            group = group.sort_values("frame_idx").reset_index(drop=True)
            features = group[self.feature_cols].values
            labels = group["attention_idx"].values

            for i in range(0, len(features) - seq_len + 1):
                seq = features[i:i + seq_len]
                label = labels[i + seq_len - 1]  # 마지막 프레임 기준 라벨
                self.sequences.append(seq)
                self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
