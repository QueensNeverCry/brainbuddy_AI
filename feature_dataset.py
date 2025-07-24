import pickle
from torch.utils.data import Dataset

class CNNFeatureDataset(Dataset):
    def __init__(self, pkl_paths):
        self.features = []
        self.labels = []

        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.features.extend(data['features'])
            self.labels.extend(data['labels'])


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx].float()
        label = self.labels[idx].long()
        return feature, label
