# inference2.py (Version 2 모델 전용 테스트)
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math  # Positional Encoding을 위해 추가

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    f1_score
)

# ------------------ Dataset (Version 2 호환) ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(img_files) >= 30:
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:30]
        frames = []
        
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            try:
                img_pil = Image.open(img_path).convert('RGB')
                frames.append(self.transform(img_pil))
            except Exception as e:
                print(f"⚠️ 이미지 로드 실패: {img_path}")
                continue
        
        # 30개 프레임 보장
        while len(frames) < 30:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        video = torch.stack(frames[:30])  # 정확히 30개만

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        # 라벨 처리 (문자열 → 숫자 변환)
        if isinstance(label, str):
            if label == '집중하지않음':
                label = 1
            else:
                label = 0  # 기본값

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Version 2 CNNEncoder ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # ✅ Version 2와 동일한 FC 레이어 (BatchNorm 추가)
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),  # BatchNorm 추가
            nn.ReLU(),
            nn.Dropout(0.4),  # 드롭아웃 증가
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):  # x: (B, 30, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

# ------------------ Version 2 Transformer 모델 ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].to(x.device)

class EngagementModelV2(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # ✅ Version 2 입력 프로젝션 (LayerNorm 포함)
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),  # LayerNorm 추가
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # ✅ Version 2 개선된 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,  # 드롭아웃 증가
            activation='gelu',  # ReLU → GELU
            batch_first=True,
            norm_first=True  # Pre-LN 구조
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ✅ Version 2 개선된 Pooling (Max + Average 조합)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # ✅ Version 2 더 복잡한 최종 분류기
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + fusion_feat_dim, 512),  # Max + Avg pooling
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        # 입력 프로젝션
        x = self.input_projection(cnn_feats)  # (B, T, d_model)
        
        # Positional Encoding 추가 (시퀀스 순서 정보)
        x = x.transpose(0, 1)  # (T, B, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, T, d_model)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (B, T, d_model)
        
        # ✅ Max + Average Pooling 조합
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, d_model * 2)
        
        # Fusion features 결합
        combined = torch.cat([pooled, fusion_feats], dim=1)  # (B, d_model * 2 + 5)
        
        # 최종 출력
        return self.fc(combined)

# ------------------ Utils ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        if not os.path.exists(pkl_path):
            print(f"⚠️ 파일을 찾을 수 없습니다: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)  # [(folder_path, label), ...]
            all_data.extend(data)
            print(f"✅ 로드됨: {pkl_path} ({len(data)}개 샘플)")
    return all_data

def test_multiple_thresholds(all_probs, all_labels):
    """여러 임계값으로 성능 테스트"""
    print("\n🎯 **임계값별 성능 비교**")
    print("="*60)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (np.array(all_probs) >= threshold).astype(np.int32)
        acc = accuracy_score(all_labels, preds)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        print(f"Threshold {threshold:.1f}: Acc={acc:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🏆 **최적 임계값: {best_threshold:.1f} (F1={best_f1:.4f})**")
    return best_threshold

# ------------------ Main Test Function ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 사용 디바이스: {device}")
    print("🚀 **Version 2 모델 테스트 시작**")
    print("="*60)

    # ✅ Version 2 모델 경로
    best_model_path = "./log/v2/best_model_v2.pt"
    
    # ✅ 테스트 데이터 경로 (더 포괄적)
    test_pkl_files = [
        
        "./preprocess2/pickle_labels/valid/20_02.pkl",  # 새로운 데이터
        "./preprocess2/pickle_labels/valid/20_04.pkl",  # 대용량 데이터
    ]

    # 데이터 로드
    test_data_list = load_data(test_pkl_files)
    if len(test_data_list) == 0:
        print("❌ 테스트 데이터가 없습니다. 경로를 확인해주세요.")
        return
    
    print(f"📊 총 테스트 데이터: {len(test_data_list):,}개")
    
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ Version 2 모델 초기화 (d_model=256, num_layers=4)
    cnn = CNNEncoder().to(device)
    model = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(device)

    if not os.path.exists(best_model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {best_model_path}")
        print("다음 경로들을 확인해보세요:")
        print("  - ./log/v2/best_model_v2.pt")
        print("  - ./log/best_model2.pt")
        return

    print(f"📂 Version 2 모델 로딩 중: {best_model_path}")
    try:
        ckpt = torch.load(best_model_path, map_location=device)
        cnn.load_state_dict(ckpt['cnn_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])
        
        # 모델 정보 출력
        if 'epoch' in ckpt:
            print(f"✅ 모델 로딩 완료 (Epoch {ckpt['epoch'] + 1}, Val Loss: {ckpt.get('val_loss', 'N/A'):.4f})")
        else:
            print("✅ 모델 로딩 완료")
            
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return

    cnn.eval()
    model.eval()

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    print("\n🔄 추론 시작...")
    with torch.no_grad():
        for videos, fusion, labels in tqdm(test_loader, desc="Version 2 Test"):
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            
            # CNN 특징 추출
            feats = cnn(videos)
            
            # Transformer 추론
            logits = model(feats, fusion)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            
            # 기본 임계값 0.5로 예측
            preds = (probs >= 0.5).astype(np.int32)
            labels = labels.int().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # ✅ 기본 성능 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("📊 **Version 2 테스트 결과 (임계값 0.5)**")
    print("="*60)
    print(f"✅ Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # 클래스별 분포 출력
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\n📋 실제 라벨 분포: {dict(zip(unique, counts))}")
    unique, counts = np.unique(all_preds, return_counts=True)
    print(f"📋 예측 라벨 분포: {dict(zip(unique, counts))}")

    # ✅ 여러 임계값으로 최적화
    best_threshold = test_multiple_thresholds(all_probs, all_labels)
    
    # 최적 임계값으로 재계산
    best_preds = (np.array(all_probs) >= best_threshold).astype(np.int32)
    best_acc = accuracy_score(all_labels, best_preds)
    best_rec = recall_score(all_labels, best_preds, zero_division=0)
    best_f1 = f1_score(all_labels, best_preds, zero_division=0)
    best_cm = confusion_matrix(all_labels, best_preds)

    print("\n" + "="*60)
    print(f"📊 **Version 2 최적화 결과 (임계값 {best_threshold:.1f})**")
    print("="*60)
    print(f"🏆 Accuracy: {best_acc:.4f} | Recall: {best_rec:.4f} | F1: {best_f1:.4f}")
    
    # ✅ 혼동행렬 저장
    save_dir = "./log/v2/test_results"
    os.makedirs(os.path.join(save_dir, "confusion_matrix"), exist_ok=True)
    
    # 기본 임계값 혼동행렬
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["집중함", "집중하지않음"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Version 2 Confusion Matrix (Threshold 0.5)")
    out_path_basic = os.path.join(save_dir, "confusion_matrix", "conf_matrix_v2_basic.png")
    plt.savefig(out_path_basic, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 최적 임계값 혼동행렬
    disp_best = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["집중함", "집중하지않음"])
    disp_best.plot(cmap=plt.cm.Blues)
    plt.title(f"Version 2 Confusion Matrix (Optimal Threshold {best_threshold:.1f})")
    out_path_best = os.path.join(save_dir, "confusion_matrix", f"conf_matrix_v2_optimal_{best_threshold:.1f}.png")
    plt.savefig(out_path_best, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Confusion matrices saved:")
    print(f"  - Basic (0.5): {out_path_basic}")
    print(f"  - Optimal ({best_threshold:.1f}): {out_path_best}")

    # ✅ 성능 비교 (Version 1과 비교용)
    print("\n" + "="*60)
    print("📈 **성능 요약**")
    print("="*60)
    print(f"🔸 Version 1 (기존): 72.5% 정확도")
    print(f"🔸 Version 2 (기본): {acc:.1%} 정확도")
    print(f"🔸 Version 2 (최적): {best_acc:.1%} 정확도")
    
    if best_acc > 0.725:
        print("🎉 Version 2가 Version 1보다 성능이 향상되었습니다!")
    elif best_acc > 0.70:
        print("✅ Version 2 성능이 양호합니다.")
    else:
        print("⚠️ Version 2에서 과적합이 발생한 것 같습니다.")

if __name__ == "__main__":
    main()
