#CNN 특징벡터 미리 추출하지 않고 end-to-end로 학습하기
import os
import pickle
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel
from sklearn.metrics import confusion_matrix, classification_report

class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, verbose=True):
        """
        data_list: List of (folder_path, label)
        """
        self.transform = transform
        self.verbose = verbose
        self.data_list = []

        for folder_path, label in data_list:
            if not os.path.isdir(folder_path):
                if self.verbose:
                    print(f"⚠️ Warning: '{folder_path}' 경로가 존재하지 않아 제외됩니다.")
                continue
            self.data_list.append((folder_path, label))

        if self.verbose:
            print(f"✅ 유효한 샘플 수: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]

        img_files = sorted([
            fname for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(img_files) < 30:
            raise ValueError(f"❌ 폴더 {folder_path}에는 이미지가 30장 이상 있어야 합니다.")

        selected_files = img_files[:30]

        frames = []
        for fname in selected_files:
            img_path = os.path.join(folder_path, fname)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        video = torch.stack(frames)  # (30, 3, 224, 224)
        return video, torch.tensor(label, dtype=torch.float32)




from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def train_or_eval(loader, cnn, model, criterion, optimizer=None, train=True, show_confusion=False):
    if train:
        cnn.train()
        model.train()
    else:
        cnn.eval()
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_logits = []

    for videos, labels in tqdm(loader, desc="Train" if train else "Valid"):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.set_grad_enabled(train):
            features = cnn(videos)
            outputs = model(features)   # raw logits
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = probs >= 0.5

        all_logits.extend(outputs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_correct += (preds == labels).sum().item()
        total_loss += loss.item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    if not train and show_confusion:
        # 출력: confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, digits=4)
        print("\n📊 Confusion Matrix:\n", cm)
        print("\n📋 Classification Report:\n", report)

        # 예시: 로짓값 & 확률 출력
        print("\n🔍 Sample Predictions:")
        for i in range(min(5, len(all_logits))):
            logit = all_logits[i][0]
            prob = 1 / (1 + np.exp(-logit))  # sigmoid
            pred = int(prob >= 0.5)
            true = int(all_labels[i][0])
            print(f"[{i}] Logit: {logit:.4f}, Prob: {prob:.4f}, Pred: {pred}, True: {true}")

    return avg_loss, accuracy

def load_multiple_pickles(pkl_paths):
    all_data = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)  # 각 파일은 [(folder_path, label), ...] 형태
    return all_data


def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 1. 여러 train/val pkl 파일 경로 지정
    train_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess/train_link.pkl"
    ]
    val_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess/val_link.pkl"
    ]

    # ✅ 2. 데이터 로드 및 통합
    train_data_list = load_multiple_pickles(train_pkl_files)
    val_data_list = load_multiple_pickles(val_pkl_files)

    # ✅ 3. Transform 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = VideoFolderDataset(train_data_list, transform, verbose=True)
    val_dataset = VideoFolderDataset(val_data_list, transform, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ✅ 4. 모델, 손실, 옵티마이저
    cnn = CNNEncoder().to(device)
    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(cnn.parameters()) + list(model.parameters()), lr=1e-4
    )

    # ✅ 5. 학습 루프
    for epoch in range(10):
        train_loss, train_acc = train_or_eval(train_loader, cnn, model, criterion, optimizer, train=True)
        val_loss, val_acc = train_or_eval(val_loader, cnn, model, criterion, train=False)

        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
