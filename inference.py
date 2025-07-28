# inference.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_dataset import CNNFeatureDataset
from models.engagement_model import EngagementModel

def load_model(model_path='best_model.pth', device='cpu'):
    model = EngagementModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device='cpu'):
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
    preds = (all_probs > 0.5).astype(int)

    # 성능 출력
    print("Classification Report:")
    print(classification_report(all_labels, preds, digits=4))

    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# 일반적인 테스트 코드
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model('best_model.pth', device=device)

#     # 테스트 데이터셋 로드
#     test_dataset = CNNFeatureDataset([
#         "./cnn_features/features/test_20_01.pkl",
#         "./cnn_features/features/test_20_03.pkl"
#     ])
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     evaluate(model, test_loader, device=device)


# DAiSEE 데이터셋의 불균형이 심함. 그래서 완화하기 위해 라벨 0인 개수만큼 test로 이용
def evaluate_with_varied_balanced_sets(model, full_dataset, device='cpu'):
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    features = np.array(full_dataset.features)
    labels = np.array(full_dataset.labels)

    # 라벨 0, 1 인덱스
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    np.random.seed(42)

    num_0 = len(idx_0)
    max_1 = len(idx_1)

    # 평가할 라벨 1 수 설정 (1x, 2x, 4x, 8x, full)
    ratios = [1, 2, 4, 8, 'full']

    for ratio in ratios:
        if ratio == 'full':
            num_1 = max_1
        else:
            num_1 = num_0 * ratio
            if num_1 > max_1:
                print(f"❗ 라벨 1 데이터가 부족하여 ratio {ratio}는 건너뜁니다.")
                continue

        print(f"\n📊 Evaluating with label 0:{num_0}, label 1:{num_1} (ratio 1:{'%.1f'%(num_1/num_0)})")

        selected_idx_1 = np.random.choice(idx_1, size=num_1, replace=False)
        selected_idx = np.concatenate([idx_0, selected_idx_1])
        np.random.shuffle(selected_idx)

        selected_features = features[selected_idx]
        selected_labels = labels[selected_idx]

        dataset = TensorDataset(
            torch.tensor(selected_features, dtype=torch.float32),
            torch.tensor(selected_labels, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        evaluate(model, loader, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('best_model.pth', device=device)

    test_dataset = CNNFeatureDataset([
        "./cnn_features/features/D_train.pkl",
        "./cnn_features/features/D_val.pkl"
    ])

    evaluate_with_varied_balanced_sets(model, test_dataset, device=device)



if __name__ == "__main__":
    main()
