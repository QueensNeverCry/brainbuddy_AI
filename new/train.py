import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os

# -------------------------------
from new.attention_dataset import AttentionDataset
from new.lstm import BaseLSTM  # <- 새 모델 (기존 BaseLSTM에서 변경)
# -------------------------------

# ✅ 하이퍼파라미터
SEQ_LEN = 30
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 5
HIDDEN_SIZE = 64
LR = 1e-3
LOG_CSV_PATH = "trainval_training_log.csv"
CLASS_LOG_CSV_PATH = "trainval_classwise_log.csv"

# ✅ seed 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 로딩 (통계 feature 포함하는 Dataset 사용)
train_dataset = AttentionDataset("C:/eye_dataset/all_features.csv", seq_len=SEQ_LEN)
valid_dataset = AttentionDataset("C:/eye_dataset/all_features_valid.csv", seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 클래스 이름
class_names = ["집중", "졸림", "집중결핍", "집중하락", "태만"]

# ✅ 모델 정의 (입력 차원 조정)
input_size = len(train_dataset.sequence_cols)       # 시계열 feature 수
stat_size = len(train_dataset.stats_cols)           # 통계 feature 수
model = BaseLSTM(input_size=input_size, stat_size=stat_size,
                      hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ✅ 로그 저장
log_rows = []
class_log_rows = []

best_val_acc = 0.0
best_model_state = None

# ✅ 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for x_seq, x_stat, y_batch in train_loader:
        x_seq = x_seq.to(device)
        x_stat = x_stat.to(device)
        y_batch = y_batch.to(device)

        # 🔍 NaN 체크
        if torch.isnan(x_seq).any() or torch.isnan(x_stat).any():
            print("❌ NaN in input (x_seq or x_stat)")
        if torch.isinf(x_seq).any() or torch.isinf(x_stat).any():
            print("❌ Inf in input (x_seq or x_stat)")

        optimizer.zero_grad()
        outputs = model(x_seq, x_stat)

        if torch.isnan(outputs).any():
            print("❌ NaN in model output")

        loss = criterion(outputs, y_batch)
        if torch.isnan(loss):
            print("❌ NaN in loss (before backward)")
            print("⚠️ Labels:", y_batch[:10])
            print("⚠️ Outputs:", outputs[:10])
            break

    print(f"📉 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

# ✅ 평가 모드
model.eval()

# Train Accuracy & Report
train_preds, train_targets = [], []

with torch.no_grad():
    for x_seq, x_stat, y_batch in train_loader:
        x_seq = x_seq.to(device)
        x_stat = x_stat.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_seq, x_stat)
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(y_batch.cpu().numpy())

train_acc = accuracy_score(train_targets, train_preds)
train_report = classification_report(train_targets, train_preds, target_names=class_names, output_dict=True)

# Valid Accuracy & Report
val_preds, val_targets = [], []

with torch.no_grad():
    for x_seq, x_stat, y_batch in val_loader:
        x_seq = x_seq.to(device)
        x_stat = x_stat.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_seq, x_stat)
        preds = torch.argmax(outputs, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_targets.extend(y_batch.cpu().numpy())

val_acc = accuracy_score(val_targets, val_preds)
val_report = classification_report(val_targets, val_preds, target_names=class_names, output_dict=True)

# ✅ 모델 저장
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_model_state = model.state_dict()
    torch.save(best_model_state, f"best_attention_lstm_trainval.pt")
    print(f"💾 Best model saved: best_attention_lstm_trainval.pt")

# ✅ 로그 기록
log_rows.append({
    "train_accuracy": train_acc,
    "valid_accuracy": val_acc,
    "train_loss": epoch_loss
})

for cls in class_names:
    class_log_rows.append({
        "class": cls,
        "train_precision": train_report[cls]["precision"],
        "train_recall": train_report[cls]["recall"],
        "train_f1": train_report[cls]["f1-score"],
        "train_support": train_report[cls]["support"],
        "val_precision": val_report[cls]["precision"],
        "val_recall": val_report[cls]["recall"],
        "val_f1": val_report[cls]["f1-score"],
        "val_support": val_report[cls]["support"]
    })

print(f"\n✅ Train Accuracy: {train_acc:.4f}")
print(f"✅ Valid Accuracy: {val_acc:.4f}")

# ✅ CSV 저장
df_log = pd.DataFrame(log_rows)
df_class = pd.DataFrame(class_log_rows)

df_log.to_csv(LOG_CSV_PATH, index=False)
df_class.to_csv(CLASS_LOG_CSV_PATH, index=False)

print(f"\n📝 훈련 로그 저장 완료: {LOG_CSV_PATH}")
print(f"📝 클래스별 성능 로그 저장 완료: {CLASS_LOG_CSV_PATH}")
