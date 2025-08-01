import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_dataset import CNNFeatureDataset
from models.simple_engagement_model import SimpleEngagementModel

def load_model(model_path='best_model.pth', device='cpu'):
    model = SimpleEngagementModel() 
    state_dict = torch.load(model_path, map_location=device)
    print("[INFO] Loaded parameter count:", len(state_dict))
    print("[INFO] Example parameter keys:", list(state_dict.keys())[:5])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device='cpu', threshold=0.65):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device).float()
            labels = labels.to(device).float().view(-1)

            outputs = model(features).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > threshold).astype(int)
    print("[INFO] Logit samples:", outputs[:10].cpu().numpy())
    print("[INFO] Sigmoid probability samples:", all_probs[:10])

    print("Classification Report:")
    print(classification_report(all_labels, preds, digits=4))

    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    plt.figure(figsize=(7, 4))
    sns.histplot(all_probs, bins=50, kde=True, color='purple')
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability (after sigmoid)")
    plt.ylabel("Count")
    plt.axvline(threshold, color='orange', linestyle='--', label=f'Threshold = {threshold}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_with_varied_balanced_sets(model, full_dataset, device='cpu', threshold=0.65):
    from torch.utils.data import TensorDataset

    features = np.array(full_dataset.features)
    labels = np.array(full_dataset.labels)

    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    np.random.seed(42)

    num_0 = len(idx_0)
    max_1 = len(idx_1)
    ratios = [1, 2, 4, 8, 'full']

    for ratio in ratios:
        if ratio == 'full':
            num_1 = max_1
        else:
            num_1 = num_0 * ratio
            if num_1 > max_1:
                print(f"[WARNING] Not enough label-1 samples for ratio {ratio}. Skipping.")
                continue

        print(f"\n[INFO] Evaluating with label 0: {num_0}, label 1: {num_1} (ratio 1:{'%.1f' % (num_1 / num_0)})")

        selected_idx_1 = np.random.choice(idx_1, size=num_1, replace=False)
        selected_idx = np.concatenate([idx_0, selected_idx_1])
        np.random.shuffle(selected_idx)

        selected_features = features[selected_idx]
        selected_labels = labels[selected_idx]

        print("[INFO] Feature stats:")
        print(" - Mean:", np.mean(selected_features))
        print(" - Std:", np.std(selected_features))
        print(" - Max:", np.max(selected_features))
        print(" - Min:", np.min(selected_features))

        dataset = TensorDataset(
            torch.tensor(selected_features, dtype=torch.float32),
            torch.tensor(selected_labels, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        evaluate(model, loader, device=device, threshold=threshold)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('best_model.pth', device=device)

    test_dataset = CNNFeatureDataset([
        "./cnn_features/features_30/D_train.pkl",
        "./cnn_features/features_30/D_val.pkl"
    ])

    evaluate_with_varied_balanced_sets(model, test_dataset, device=device, threshold=0.65)

if __name__ == "__main__":
    main()