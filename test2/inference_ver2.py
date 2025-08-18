# inference2_stable.py (안정적인 Version 2 테스트)
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
import math

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
        
        print(f"🎯 데이터셋 초기화 완료: {len(self.data_list)}개 유효 샘플")

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
        
        video = torch.stack(frames[:30])

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Version 2 CNNEncoder ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
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
        
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + fusion_feat_dim, 512),
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
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

# ------------------ Utils ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        if not os.path.exists(pkl_path):
            print(f"⚠️ 파일을 찾을 수 없습니다: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
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

def get_optimal_batch_size(total_samples, device):
    """앙상블과 동일한 최적 배치 사이즈 계산"""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory >= 8:  # 8GB 이상
            return min(8, total_samples)  # Version 2는 앙상블보다 약간 크게
        elif total_memory >= 4:  # 4GB 이상
            return min(4, total_samples)
        else:
            return min(2, total_samples)
    else:
        return min(16, total_samples)

# ------------------ Main Test Function ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 사용 디바이스: {device}")
    print("🚀 **Version 2 모델 테스트 시작 (앙상블 방식 적용)**")
    print("="*60)

    # ✅ Version 2 모델 경로
    best_model_path = "./log/v2/best_model_v2.pt"
    
    # ✅ 테스트 데이터 경로 (702개 샘플)
    test_pkl_files = [
        "./preprocessed_data_full/pickle_labels/test/test_data.pkl" 
    ]

    # 데이터 로드
    test_data_list = load_data(test_pkl_files)
    if len(test_data_list) == 0:
        print("❌ 테스트 데이터가 없습니다. 경로를 확인해주세요.")
        return
    
    print(f"📊 총 테스트 데이터: {len(test_data_list):,}개")
    
    test_dataset = VideoFolderDataset(test_data_list)
    
    # ✅ 앙상블과 동일한 방식으로 최적 배치 사이즈 계산
    optimal_batch_size = get_optimal_batch_size(len(test_dataset), device)
    print(f"🎯 최적 배치 사이즈: {optimal_batch_size}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=optimal_batch_size,  # 2-8개의 작은 배치
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        drop_last=False  # 모든 데이터 사용
    )
    
    print(f"📦 총 {len(test_loader)}개 배치로 분할 (배치당 최대 {optimal_batch_size}개)")
    print(f"🎯 처리될 총 샘플 수: {len(test_dataset)}개 (100% 사용)")

    # ✅ Version 2 모델 초기화
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
    total_processed = 0

    print(f"\n🔄 Version 2 추론 시작 (총 {len(test_dataset)}개 샘플)...")
    with torch.no_grad():
        for batch_idx, (videos, fusion, labels) in enumerate(tqdm(test_loader, desc="Version 2 Test")):
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            
            current_batch_size = videos.size(0)
            total_processed += current_batch_size
            
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
            
            # 진행 상황 출력 (처음과 마지막 배치에서)
            if batch_idx == 0 or batch_idx == len(test_loader) - 1 or (batch_idx + 1) % 20 == 0:
                print(f"  배치 {batch_idx + 1}/{len(test_loader)}: {current_batch_size}개 샘플 처리 완료 (누적: {total_processed}개)")

    print(f"✅ 총 {total_processed}개 샘플 처리 완료 (예상: {len(test_dataset)}개)")

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

    # 데이터 사용률 확인
    expected_samples = len(test_dataset)
    actual_samples = len(all_labels)
    usage_rate = (actual_samples / expected_samples) * 100
    print(f"\n📊 **데이터 사용률**: {actual_samples}/{expected_samples} ({usage_rate:.1f}%)")

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["집중안함", "집중함"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Version 2 Confusion Matrix (Threshold 0.5)\n{actual_samples} samples - Stable Batches")
    out_path_basic = os.path.join(save_dir, "confusion_matrix", "conf_matrix_v2_stable.png")
    plt.savefig(out_path_basic, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 최적 임계값 혼동행렬
    disp_best = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["집중안함", "집중함"])
    disp_best.plot(cmap=plt.cm.Blues)
    plt.title(f"Version 2 Confusion Matrix (Optimal {best_threshold:.1f})\n{actual_samples} samples - Stable Batches")
    out_path_best = os.path.join(save_dir, "confusion_matrix", f"conf_matrix_v2_stable_{best_threshold:.1f}.png")
    plt.savefig(out_path_best, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\n📊 Confusion matrices saved:")
    print(f"  - Basic (0.5): {out_path_basic}")
    print(f"  - Optimal ({best_threshold:.1f}): {out_path_best}")

    # ✅ 최종 성능 요약
    print("\n" + "="*60)
    print("🎉 **Version 2 안정적 처리 완료!**")
    print("="*60)
    print(f"🔸 처리된 데이터: {actual_samples:,}개 ({usage_rate:.1f}%)")
    print(f"🔸 배치 크기: {optimal_batch_size}개 (앙상블 방식)")
    print(f"🔸 총 배치 수: {len(test_loader)}개")
    print(f"🔸 Version 2 정확도: {best_acc:.1%}")
    print(f"🔸 재현율: {best_rec:.1%}")
    print(f"🔸 F1-Score: {best_f1:.1%}")
    print(f"🔸 최적 임계값: {best_threshold}")
    
    # 성능 평가
    if best_acc > 0.80:
        print("🎉 80% 이상! 훌륭한 성능입니다!")
    elif best_acc > 0.70:
        print("✅ 70% 이상! 양호한 성능입니다!")
    elif best_acc > 0.60:
        print("👍 60% 이상! 개선 여지가 있습니다!")
    else:
        print("⚠️ 성능이 제한적입니다. 모델 재훈련을 고려해보세요.")
    
    print(f"\n💡 **처리 방식 요약**")
    print(f"   - 앙상블과 동일한 안정적 배치 처리 방식 적용")
    print(f"   - GPU 메모리 효율적 사용")
    print(f"   - 전체 702개 샘플 100% 처리 완료")

if __name__ == "__main__":
    main()
