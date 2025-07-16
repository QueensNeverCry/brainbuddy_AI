# video_engagement_feature_dataset.py
import os
import torch
from torch.utils.data import Dataset
import glob

class VideoEngagementFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = sorted(glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        data = torch.load(sample_path, map_location='cpu', weights_only=True)
        return data['features'].float(), data['label']
