#CNN 특징벡터 미리 추출하지 않고 end-to-end로 학습하는 코드입니다.
# 1. VideoFolderDataset
# 2. 각 파일은 [(folder_path, label), ...] 형태로 pkl 파일에 저장되어 있습니다.(pkl 경로 : pickle_labels 안에 있습니다)
#    따라서 해당 경로를 읽으면 그 안에 30프레임이 들어있는 구조입니다.
# 3. Optimizer : Adam, 
#    loss : BCEWithLogitsLoss, 
#    Scheduler : ReduceLROnPlateau (F1-score가 향상되지 않으면 LR을 0.5배 감소)
# 4. 중간 체크포인트 저장 (매 epoch마다) : checkpoint_fold{n}.pth
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import csv
from torch.utils.data import WeightedRandomSampler

# Pytorch Dataset 객체 정의
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
                # if self.verbose:
                #     print(f"⚠️ Warning: '{folder_path}' 경로가 존재하지 않아 제외됩니다.")
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


def evaluate_and_visualize(y_true, y_probs, epoch=None, save_dir="visualizations"):
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_threshold = 0.0, 0.5
    best_preds = (y_probs >= 0.5).astype(int)

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average='weighted')
        if f1 > best_f1:
            best_f1, best_threshold, best_preds = f1, t, preds

    acc = (best_preds == y_true).mean()
    auc_score = roc_auc_score(y_true, y_probs)

    # 🔹 Confusion matrix
    cm = confusion_matrix(y_true, best_preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    plt.savefig(os.path.join(save_dir, f"conf_matrix_epoch{epoch+1}.png"))
    plt.close()

    # 🔹 ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch+1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_curve_epoch{epoch+1}.png"))
    plt.close()

    return best_preds, best_threshold, best_f1, acc, auc_score



def train_or_eval(loader, cnn, model, criterion, optimizer=None, train=True, show_confusion=False, accumulation_steps=16, threshold=0.5):
    if train:
        cnn.train()
        model.train()
        optimizer.zero_grad()
    else:
        cnn.eval()
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_logits = []

    for step, (videos, labels) in enumerate(tqdm(loader, desc="Train" if train else "Valid")):
        videos = videos.to(device)
        labels = labels.to(device).unsqueeze(1)

        with torch.set_grad_enabled(train):
            features = cnn(videos)
            outputs = model(features)
            probs = torch.sigmoid(outputs)

            if not train and step == 0:
                print("🔍 outputs:", outputs.squeeze().tolist())
                print("🔍 probs:", probs.squeeze().tolist())
                print("🔍 labels:", labels.squeeze().tolist())

            loss = criterion(outputs, labels)

            if train:
                loss = loss / accumulation_steps
                loss.backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()

        # 예측 및 저장
        preds = probs >= threshold
        all_logits.extend(outputs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * accumulation_steps
        total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    if not train and show_confusion:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, digits=4)
        print("\n📊 Confusion Matrix:\n", cm)
        print("\n📋 Classification Report:\n", report)

        print("\n🔍 Sample Predictions:")
        for i in range(min(5, len(all_logits))):
            logit = all_logits[i][0]
            prob = 1 / (1 + np.exp(-logit))
            pred = int(prob >= 0.5)
            true = int(all_labels[i][0])
            print(f"[{i}] Logit: {logit:.4f}, Prob: {prob:.4f}, Pred: {pred}, True: {true}")

    return avg_loss, accuracy

#(폴더경로, 라벨)이 담긴 pkl파일 읽기
def load_multiple_pickles(pkl_paths):
    all_data = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)  
    return all_data

def main(resume_only=True):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 결과 csv 저장 경로
    log_path = "log/training_log.csv"
    os.makedirs("log", exist_ok=True)
    first_write = not os.path.exists(log_path) 
    if first_write:
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "f1_score", "best_thresh", "auc"])


    # train/val pkl 로드
    train_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl",
        #"C:/KSEB/brainbuddy_AI/preprocess/train_link.pkl"
    ]
    val_pkl_files = [
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
        "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_03.pkl",
        #"C:/KSEB/brainbuddy_AI/preprocess/val_link.pkl"
    ]

    train_data_list = load_multiple_pickles(train_pkl_files)
    val_data_list = load_multiple_pickles(val_pkl_files)

    print(f"📦 Train 샘플 수: {len(train_data_list)}")# 데이터 샘플 수 및 분포 비율 출력
    print(f"📦 Valid 샘플 수: {len(val_data_list)}")
    train_labels = [label for _, label in train_data_list]
    val_labels = [label for _, label in val_data_list]
    train_pos_ratio = np.mean(train_labels)
    val_pos_ratio = np.mean(val_labels)
    print(f"📊 Train 클래스 분포: 1 비율 = {train_pos_ratio:.4f}, 0 비율 = {1 - train_pos_ratio:.4f}")
    print(f"📊 Valid 클래스 분포: 1 비율 = {val_pos_ratio:.4f}, 0 비율 = {1 - val_pos_ratio:.4f}")
 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # DAiSEE만 써서 학습해보았을 때: weightedRandomSampler로 불균형 보정
    # class_counts = np.bincount(train_labels)
    # weights = 1. / class_counts
    # sample_weights = [weights[label] for _, label in train_data_list]
    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


    train_dataset = VideoFolderDataset(train_data_list, transform=transform, verbose=False)
    val_dataset = VideoFolderDataset(val_data_list, transform=transform, verbose=False)
    #train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,num_workers=8, pin_memory=True)

    # 모델/옵티마이저 정의
    cnn = CNNEncoder().to(device)
    model = EngagementModel().to(device)
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    # 체크포인트가 있다면 로드해서 이어서 학습하기
    checkpoint_path = "checkpoint.pth"
    best_model_path = "best_model.pth"
    start_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0

    if resume_only and os.path.exists(checkpoint_path):
        print(f"🔁 체크포인트 로드됨: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(checkpoint["cnn_state_dict"])
        model.load_state_dict(checkpoint["lstm_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        best_val_f1 = checkpoint["best_val_f1"]

    accumulation_steps = 16

    # train
    for epoch in range(start_epoch, 20):
        # F1-score 계산
        cnn.eval()
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(cnn(videos))
                probs = torch.sigmoid(outputs)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 동적 threshold 평가
        final_preds, best_thresh, val_f1, val_acc, val_auc= evaluate_and_visualize(
            y_true=all_labels, 
            y_probs=np.array(all_probs), 
            epoch=epoch
        )
        train_loss, train_acc = train_or_eval(train_loader, cnn, model, criterion, optimizer, train=True,accumulation_steps=accumulation_steps,threshold=best_thresh)
        val_loss, val_acc = train_or_eval(val_loader, cnn, model, criterion, train=False,accumulation_steps=accumulation_steps,threshold=best_thresh)

        scheduler.step(val_f1) 

        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Best Thresh: {best_thresh:.2f}")

        
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(train_loss, 4),
                round(train_acc, 4),
                round(val_loss, 4),
                round(val_acc, 4),
                round(val_f1, 4),
                round(best_thresh, 4),
                round(val_auc, 4)
            ])

        # 최고 성능 모델 저장
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                "cnn_state_dict": cnn.state_dict(),
                "lstm_state_dict": model.state_dict(),
            }, best_model_path)
            print("✅ 최고 성능 모델 저장됨")

        # 체크포인트 저장
        torch.save({
            "epoch": epoch + 1,
            "cnn_state_dict": cnn.state_dict(),
            "lstm_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1
        }, checkpoint_path)
        print("💾 체크포인트 저장 완료")



if __name__ == "__main__":
    main(resume_only=True)
