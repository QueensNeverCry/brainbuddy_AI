import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



train_paths = [
    "./cnn_features/features/train_20_01.pkl",
    "./cnn_features/features/train_20_03.pkl",
    "./cnn_features/features/D_train.pkl",
    "./cnn_features/features/eng.pkl"
]
val_paths = [
    "./cnn_features/features/valid_20_01.pkl",
    "./cnn_features/features/valid_20_03.pkl",
    "./cnn_features/features/D_val.pkl"
]

def load_features(paths):
    all_features = []
    all_labels = []
    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            features = data['features']  # list of (T, 1280)
            labels = data['labels']      # list of 0/1
            all_features.extend(features)
            all_labels.extend(labels)
    return np.array(all_features), np.array(all_labels)


# 🔹 데이터 로딩
features_train, labels_train = load_features(train_paths)
features_val, labels_val = load_features(val_paths)

print(f"Train feature shape: {features_train.shape}")  # (B, T, 1280)
print(f"Validation feature shape: {features_val.shape}")

# 🔹 Time 평균 pooling (LSTM 없이)
X_train = features_train.mean(axis=1)  # (B, 1280)
X_val = features_val.mean(axis=1)

# 🔹 Logistic Regression 분류기 학습
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, labels_train)

# 🔹 평가
y_pred = clf.predict(X_val)

print("\n[Classification Report]")
print(classification_report(labels_val, y_pred))

# 🔹 Confusion Matrix 시각화
cm = confusion_matrix(labels_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
