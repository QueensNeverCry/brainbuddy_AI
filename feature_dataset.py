import pickle
import torch
from torch.utils.data import Dataset

class CNNFeatureDataset(Dataset):
    def __init__(self, pickle_files):
        self.features = []
        self.labels = []
        
        for pkl_path in pickle_files:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)  # data는 {'Engaged': [...], 'Not engaged': [...]} 형식
                
            for label_name, samples in data.items():
                label = 1 if label_name == 'Engaged' else 0
                for sample in samples:
                    # sample이 list 형태면 tensor로 변환
                    if not torch.is_tensor(sample):
                        sample = torch.tensor(sample)
                    self.features.append(sample)
                    self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
