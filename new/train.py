import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os

# -------------------------------
# Dataset 클래스 불러오기
from your_module import AttentionDataset, AttentionLSTM
# -------------------------------

# ✅ 하이퍼파라미터
SEQ_LEN = 30
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 5
NUM_FOLDS = 5
HIDDEN_SIZE = 64
LR = 1e-3
LOG_CSV_PATH = "lstm_training_log.csv"

# ✅ seed 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ✅ 로그 저장용 리스트
log_rows = []

# ✅ 데이터셋 전체 로딩
dataset = AttentionDataset("C:/eye_dataset/all_features.csv", seq_len=SEQ_LEN)
X = np.arange(len(dataset))
y = np.array(dataset.labels)

# ✅ K-Fold 설정
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🔁 Fold {fold+1}/{NUM_FOLDS}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionLSTM(input_size=len(dataset.feature_cols), hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
    model = model.cuda() if torch.cuda.is_available() else model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ✅ Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda() if torch.cuda.is_available() else x_batch
            y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"  📉 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

    # ✅ Train Accuracy 계산
    model.eval()
    train_preds, train_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda() if torch.cuda.is_available() else x_batch
            outputs = model(x_batch)
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(y_batch.numpy())

    train_acc = accuracy_score(train_targets, train_preds)

    # ✅ Validation Accuracy 계산
    val_preds, val_targets = [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.cuda() if torch.cuda.is_available() else x_val
            outputs = model(x_val)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_targets.extend(y_val.numpy())

    val_acc = accuracy_score(val_targets, val_preds)
    fold_accuracies.append(val_acc)

    # ✅ 로그 기록
    log_rows.append({
        "fold": fold + 1,
        "train_accuracy": train_acc,
        "valid_accuracy": val_acc,
        "train_loss": epoch_loss
    })

    print(f"✅ Fold {fold+1} Train Accuracy: {train_acc:.4f}")
    print(f"✅ Fold {fold+1} Valid Accuracy: {val_acc:.4f}")

# ✅ 전체 평균 정확도
print("\n📊 K-Fold 완료")
print(f"Fold Accuracies: {fold_accuracies}")
print(f"📈 평균 Accuracy: {np.mean(fold_accuracies):.4f}")

# ✅ CSV 저장
df_log = pd.DataFrame(log_rows)
df_log.to_csv(LOG_CSV_PATH, index=False)
print(f"\n📝 훈련 로그 저장 완료: {LOG_CSV_PATH}")
