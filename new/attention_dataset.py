import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AttentionDataset(Dataset):
    def __init__(self, csv_path, seq_len=30):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len

        # 사용할 feature 열 (라벨 제외)
        self.feature_cols = [col for col in self.df.columns 
                             if col not in ["folder_name", "frame_idx", "attention_label", "attention_idx"]]

        # 그룹 단위 (폴더별 시퀀스)
        self.sequences = []
        self.labels = []

        grouped = self.df.groupby("folder_name")
        for name, group in grouped:
            group = group.sort_values("frame_idx").reset_index(drop=True)
            features = group[self.feature_cols].values
            label = group["attention_idx"].iloc[0]  # 같은 폴더는 동일 라벨로 가정

            # 시퀀스를 일정 길이로 자름
            for i in range(0, len(features) - seq_len + 1):
                seq = features[i:i + seq_len]
                self.sequences.append(seq)
                self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
