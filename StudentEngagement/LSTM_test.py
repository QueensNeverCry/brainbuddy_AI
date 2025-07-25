import torch
import torch.nn as nn
import numpy as np
from feature_dataset import CNNFeatureDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out * weights, dim=1)
        return context

class EngagementModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, output_size=1):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        context = self.attn(lstm_out)
        context = self.norm(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    test_feature_path = os.path.normpath(r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/test/test_features_aug.pkl")
    model_path = os.path.normpath(r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/best_model.pth")

    test_dataset = CNNFeatureDataset([test_feature_path])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=2)

    model = EngagementModel(input_size=512).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1)

            outputs = model(features).squeeze(1)
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds = (all_probs > 0.5).astype(int)

    f1 = f1_score(all_labels, preds)
    print(f"\n✅ Test F1 Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")
    plt.show()

    plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label="Positive")
    plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label="Negative")
    plt.xlabel("Sigmoid Output")
    plt.ylabel("Frequency")
    plt.title("Test Sigmoid Probability Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
