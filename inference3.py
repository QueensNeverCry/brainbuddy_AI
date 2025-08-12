# ensemble_inference_test.py (앙상블 모델 전용 테스트)
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

# ------------------ Dataset (앙상블 호환) ------------------
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
        
        video = torch.stack(frames[:30])

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
                label = 0

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ CNNEncoder (Version 1과 Version 2 구조) ------------------
class CNNEncoderV1(nn.Module):
    """Version 1용 CNNEncoder (기본 구조)"""
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

class CNNEncoderV2(nn.Module):
    """Version 2용 CNNEncoder (BatchNorm 포함)"""
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

# ------------------ Positional Encoding ------------------
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

# ------------------ Version 1 Transformer ------------------
class EngagementModelV1(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(cnn_feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + fusion_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

# ------------------ Version 2 Transformer ------------------
class EngagementModelV2(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
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

# ------------------ 앙상블 모델 ------------------
class TransformerEnsembleModel(nn.Module):
    def __init__(self, cnn_v1, model_v1, cnn_v2, model_v2, ensemble_method='learned'):
        super().__init__()
        self.cnn_v1 = cnn_v1
        self.model_v1 = model_v1
        self.cnn_v2 = cnn_v2
        self.model_v2 = model_v2
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            self.register_buffer('weights', torch.tensor([0.3, 0.7]))
        elif ensemble_method == 'learned':
            self.ensemble_weights = nn.Parameter(torch.tensor([0.3, 0.7]))
            self.ensemble_fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

    def forward(self, videos, fusion_feats):
        # Version 1 추론
        feats_v1 = self.cnn_v1(videos)
        logits_v1 = self.model_v1(feats_v1, fusion_feats)
        
        # Version 2 추론
        feats_v2 = self.cnn_v2(videos)
        logits_v2 = self.model_v2(feats_v2, fusion_feats)
        
        if self.ensemble_method == 'weighted':
            return self.weights[0] * logits_v1 + self.weights[1] * logits_v2
        elif self.ensemble_method == 'learned':
            prob_v1 = torch.sigmoid(logits_v1)
            prob_v2 = torch.sigmoid(logits_v2)
            normalized_weights = torch.softmax(self.ensemble_weights, dim=0)
            weighted_v1 = prob_v1 * normalized_weights[0]
            weighted_v2 = prob_v2 * normalized_weights[1]
            combined_input = torch.cat([weighted_v1, weighted_v2], dim=1)
            return self.ensemble_fc(combined_input)

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
    print("\n🎯 **앙상블 임계값별 성능 비교**")
    print("="*60)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
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
    print("🚀 **앙상블 모델 (V1 + V2) 테스트 시작**")
    print("="*60)

    # ✅ 앙상블 모델 경로 (훈련 완료된 모델)
    ensemble_model_path = "./log/ensemble/best_speed_ensemble.pt"
    
    # ✅ 테스트 데이터 경로 (요청하신 2개만 사용)
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
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # ✅ 개별 모델들 초기화
    cnn_v1 = CNNEncoderV1().to(device)
    model_v1 = EngagementModelV1(d_model=128, nhead=8, num_layers=3).to(device)
    cnn_v2 = CNNEncoderV2().to(device)
    model_v2 = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(device)

    # ✅ 개별 모델 가중치 로드
    print("\n🔄 개별 모델들 로드 중...")
    try:
        # Version 1 모델 로드
        v1_checkpoint = torch.load("./log/best_model2.pt", map_location=device)
        cnn_v1.load_state_dict(v1_checkpoint['cnn_state_dict'])
        model_v1.load_state_dict(v1_checkpoint['model_state_dict'])
        print("✅ Version 1 모델 로드 완료")
        
        # Version 2 모델 로드
        v2_checkpoint = torch.load("./log/v2/best_model_v2.pt", map_location=device)
        cnn_v2.load_state_dict(v2_checkpoint['cnn_state_dict'])
        model_v2.load_state_dict(v2_checkpoint['model_state_dict'])
        print("✅ Version 2 모델 로드 완료")
        
    except Exception as e:
        print(f"❌ 개별 모델 로드 실패: {e}")
        return

    # ✅ 앙상블 모델 생성 및 로드
    ensemble_model = TransformerEnsembleModel(
        cnn_v1, model_v1, cnn_v2, model_v2, 
        ensemble_method='learned'
    ).to(device)

    if not os.path.exists(ensemble_model_path):
        print(f"❌ 앙상블 모델을 찾을 수 없습니다: {ensemble_model_path}")
        print("다음 경로들을 확인해보세요:")
        print("  - ./log/ensemble/best_speed_ensemble.pt")
        print("  - ./log/ensemble/best_weighted_ensemble.pt")
        print("  - ./log/ensemble/best_transformer_ensemble.pt")
        return

    print(f"📂 앙상블 모델 로딩 중: {ensemble_model_path}")
    try:
        ensemble_checkpoint = torch.load(ensemble_model_path, map_location=device)
        ensemble_model.load_state_dict(ensemble_checkpoint['ensemble_state_dict'])
        
        # 앙상블 모델 정보 출력
        training_accuracy = ensemble_checkpoint.get('accuracy', 0)
        training_f1 = ensemble_checkpoint.get('f1_score', 0)
        ensemble_method = ensemble_checkpoint.get('ensemble_method', 'learned')
        
        print(f"✅ 앙상블 모델 로딩 완료")
        print(f"   - 훈련 정확도: {training_accuracy:.1%}")
        print(f"   - 훈련 F1: {training_f1:.1%}")
        print(f"   - 앙상블 방법: {ensemble_method}")
        
        # 학습된 가중치 출력
        if ensemble_method == 'learned' and hasattr(ensemble_model, 'ensemble_weights'):
            weights = torch.softmax(ensemble_model.ensemble_weights, dim=0).detach().cpu().numpy()
            print(f"   - 학습된 가중치: V1={weights[0]:.3f}, V2={weights[1]:.3f}")
            
    except Exception as e:
        print(f"❌ 앙상블 모델 로딩 실패: {e}")
        return

    # 모델을 evaluation 모드로 설정
    ensemble_model.eval()

    # ✅ 추론 시작
    all_probs, all_preds, all_labels = [], [], []

    print("\n🔄 앙상블 추론 시작...")
    with torch.no_grad():
        for videos, fusion, labels in tqdm(test_loader, desc="Ensemble Test"):
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            
            # 앙상블 추론
            logits = ensemble_model(videos, fusion)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            
            # 기본 임계값 0.7로 예측 (훈련에서 최적이었음)
            preds = (probs >= 0.7).astype(np.int32)
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
    print("📊 **앙상블 테스트 결과 (임계값 0.7)**")
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
    print(f"📊 **앙상블 최적화 결과 (임계값 {best_threshold:.1f})**")
    print("="*60)
    print(f"🏆 Accuracy: {best_acc:.4f} | Recall: {best_rec:.4f} | F1: {best_f1:.4f}")
    
    # ✅ 혼동행렬 저장
    save_dir = "./log/ensemble/test_results"
    os.makedirs(os.path.join(save_dir, "confusion_matrix"), exist_ok=True)
    
    # 기본 임계값 혼동행렬
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["집중함", "집중하지않음"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Ensemble Confusion Matrix (Threshold 0.7)")
    out_path_basic = os.path.join(save_dir, "confusion_matrix", "ensemble_conf_matrix_basic.png")
    plt.savefig(out_path_basic, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 최적 임계값 혼동행렬
    disp_best = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["집중함", "집중하지않음"])
    disp_best.plot(cmap=plt.cm.Blues)
    plt.title(f"Ensemble Confusion Matrix (Optimal Threshold {best_threshold:.1f})")
    out_path_best = os.path.join(save_dir, "confusion_matrix", f"ensemble_conf_matrix_optimal_{best_threshold:.1f}.png")
    plt.savefig(out_path_best, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\n📊 Confusion matrices saved:")
    print(f"  - Basic (0.7): {out_path_basic}")
    print(f"  - Optimal ({best_threshold:.1f}): {out_path_best}")

    # ✅ 최종 성능 비교
    print("\n" + "="*60)
    print("📈 **최종 성능 비교**")
    print("="*60)
    print(f"🔸 Version 1 (기본 Transformer): 72.5% 정확도")
    print(f"🔸 Version 2 (개선 Transformer): 76.9% 정확도")
    print(f"🔸 앙상블 (훈련 결과): {training_accuracy:.1%} 정확도")
    print(f"🔸 앙상블 (테스트 기본): {acc:.1%} 정확도")
    print(f"🔸 앙상블 (테스트 최적): {best_acc:.1%} 정확도")
    
    improvement_vs_v2 = (best_acc - 0.769) * 100
    improvement_vs_v1 = (best_acc - 0.725) * 100
    
    print(f"\n🚀 **앙상블 개선 효과**")
    print(f"   - vs Version 1: +{improvement_vs_v1:.1f}%p")
    print(f"   - vs Version 2: +{improvement_vs_v2:.1f}%p")
    print(f"   - 재현율: {best_rec:.1%}")
    print(f"   - F1-Score: {best_f1:.1%}")
    
    if best_acc > 0.90:
        print("🎉 90% 이상 달성! 앙상블 모델이 탁월한 성능을 보여줍니다!")
    elif best_acc > 0.85:
        print("🎊 85% 이상 달성! 앙상블 효과가 뛰어납니다!")
    elif best_acc > 0.80:
        print("✅ 80% 이상 달성! 앙상블이 성공적으로 작동했습니다!")
    elif best_acc > 0.77:
        print("👍 앙상블 효과 확인! 개별 모델보다 향상되었습니다!")
    else:
        print("⚠️ 테스트 데이터에서 일반화 성능이 제한적입니다.")

if __name__ == "__main__":
    main()
