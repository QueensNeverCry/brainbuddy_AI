# precision_enhanced_v2.py (Precision 0.9+ 달성을 위한 Version 2 개선)
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score, precision_score
import math
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time

# ------------------ Precision 최적화 Dataset ------------------
class PrecisionEnhancedDataset(Dataset):
    def __init__(self, data_list, transform=None, is_training=True):
        self.data_list = []
        self.is_training = is_training
        
        if is_training:
            # 더 강한 데이터 증강으로 robust한 특징 학습
            self.transform = transform or transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # 덜 aggressive한 크롭
                transforms.RandomHorizontalFlip(p=0.3),  # 확률 줄임
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),  # 약한 색상 변화
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # 데이터 품질 확인을 더 엄격하게
        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(img_files) >= 30:  # 충분한 프레임 수 확보
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:30]
        frames = []
        
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            try:
                img_pil = Image.open(img_path).convert('RGB')
                frames.append(self.transform(img_pil))
            except Exception as e:
                continue
        
        # 30개 프레임 보장
        while len(frames) < 30:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        video = torch.stack(frames[:30])

        # Fusion features
        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Precision 최적화 CNNEncoder ------------------
class PrecisionEnhancedCNNEncoder(nn.Module):
    """Precision 향상을 위한 더 robust한 CNN 인코더"""
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 더 깊고 정교한 FC 레이어
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.25),  # 드롭아웃 줄여서 overfitting 방지하면서 정확도 유지
            
            nn.Linear(2048, 1280),
            nn.BatchNorm1d(1280),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(1280, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
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

# ------------------ Precision 강화 Loss ------------------
class PrecisionFocusedLoss(nn.Module):
    """Precision 향상에 특화된 손실 함수 (완전한 CUDA 오류 해결)"""
    def __init__(self, precision_weight=2.0, pos_weight=1.5):
        super().__init__()
        self.precision_weight = precision_weight
        self.pos_weight_value = pos_weight  # 값만 저장
        
    def forward(self, logits, labels):
        # 매번 현재 디바이스에 맞춰 pos_weight 생성
        device = logits.device
        pos_weight = torch.tensor([self.pos_weight_value], device=device, dtype=logits.dtype)
        
        # BCE Loss 생성
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        bce = bce_loss(logits, labels)
        
        # Precision 집중: False Positive 페널티
        probs = torch.sigmoid(logits)
        false_positives = (1 - labels) * probs
        fp_penalty = torch.mean(false_positives) * self.precision_weight
        
        return bce + fp_penalty

# ------------------ Precision 최적화 모델 ------------------
class PrecisionEnhancedModelV2(nn.Module):
    """Precision 0.9+ 달성을 위한 개선된 Version 2"""
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # 더 robust한 입력 프로젝션
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 개선된 Transformer (더 깊고 안정적)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.12,  # 적당한 드롭아웃
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 다중 스케일 풀링
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention 메커니즘 (중요한 프레임에 집중)
        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Precision 최적화를 위한 보수적 분류기
        self.classifier = nn.Sequential(
            # 첫 번째 블록
            nn.Linear(d_model * 3 + fusion_feat_dim, 512),  # attention pooling 추가로 d_model * 3
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # 두 번째 블록
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            # 세 번째 블록 (더 보수적)
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # 최종 출력
            nn.Linear(128, 1)
        )
        
        # Precision을 위한 확신도 조절 파라미터
        self.confidence_scaling = nn.Parameter(torch.tensor([1.2]))  # 학습 가능한 스케일링

    def forward(self, cnn_feats, fusion_feats):
        # CNN 특징 프로젝션
        x = self.input_projection(cnn_feats)  # (B, T, d_model)
        
        # Positional Encoding
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer 처리
        transformer_out = self.transformer_encoder(x)  # (B, T, d_model)
        
        # 다중 풀링
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        
        # Attention 기반 풀링 (중요한 프레임에 집중)
        attention_scores = self.attention_weights(transformer_out)  # (B, T, 1)
        attention_pooled = torch.sum(transformer_out * attention_scores, dim=1)  # (B, d_model)
        
        # 모든 풀링 결합
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # (B, d_model * 3)
        
        # Fusion features와 결합
        combined = torch.cat([pooled_features, fusion_feats], dim=1)
        
        # 분류
        logits = self.classifier(combined)
        
        # Precision 최적화: 확신도 스케일링 (더 보수적 예측)
        scaled_logits = logits * self.confidence_scaling
        
        return scaled_logits

# ------------------ 훈련 및 검증 함수 ------------------
def train_precision_model(model, cnn, loader, criterion, optimizer, device, scaler):
    model.train()
    cnn.train()
    total_loss = 0

    for videos, fusion, labels in tqdm(loader, desc="Precision-Focused Training"):
        videos = videos.to(device, non_blocking=True)
        fusion = fusion.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        
        with autocast():
            cnn_features = cnn(videos)
            logits = model(cnn_features, fusion)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    return total_loss / len(loader)

def validate_precision_model(model, cnn, loader, criterion, device):
    model.eval()
    cnn.eval()
    total_loss = 0

    with torch.no_grad():
        for videos, fusion, labels in tqdm(loader, desc="Precision Validation"):
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            with autocast():
                cnn_features = cnn(videos)
                logits = model(cnn_features, fusion)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_precision_metrics(model, cnn, loader, device, threshold=0.8):
    """Precision 중심 평가"""
    model.eval()
    cnn.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, fusion, labels in loader:
            videos = videos.to(device)
            fusion = fusion.to(device)
            
            cnn_features = cnn(videos)
            logits = model(cnn_features, fusion)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.int().numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return accuracy, precision, recall, f1, all_probs

def find_optimal_threshold_for_precision(all_probs, all_labels, target_precision=0.9):
    """Precision 0.9+ 달성을 위한 최적 임계값 탐색"""
    print("\n🎯 Precision 0.9+ 달성을 위한 임계값 최적화")
    print("="*60)
    
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in np.arange(0.5, 0.95, 0.02):  # 0.5부터 0.95까지 세밀하게
        preds = (np.array(all_probs) >= threshold).astype(int)
        
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        accuracy = accuracy_score(all_labels, preds)
        
        print(f"Threshold {threshold:.2f}: Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f} | Acc={accuracy:.3f}")
        
        # Precision 0.9+ 달성하는 첫 번째 임계값 선택
        if precision >= target_precision and not best_metrics:
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
    
    if best_metrics:
        print(f"\n🏆 Precision {target_precision}+ 달성!")
        print(f"   - 최적 임계값: {best_metrics['threshold']:.2f}")
        print(f"   - Precision: {best_metrics['precision']:.3f}")
        print(f"   - Recall: {best_metrics['recall']:.3f}")
        print(f"   - F1-Score: {best_metrics['f1']:.3f}")
        print(f"   - Accuracy: {best_metrics['accuracy']:.3f}")
        return best_metrics['threshold'], best_metrics
    else:
        print(f"⚠️ Precision {target_precision} 미달성")
        # 가장 높은 Precision 반환
        best_precision = 0
        fallback_threshold = 0.8
        for threshold in np.arange(0.5, 0.95, 0.02):
            preds = (np.array(all_probs) >= threshold).astype(int)
            precision = precision_score(all_labels, preds, zero_division=0)
            if precision > best_precision:
                best_precision = precision
                fallback_threshold = threshold
        
        print(f"📊 최고 달성 Precision: {best_precision:.3f} (임계값: {fallback_threshold:.2f})")
        return fallback_threshold, None

def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    
    import random
    random.shuffle(all_data)
    return all_data

# ------------------ 메인 훈련 함수 ------------------
def main():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 **Precision 0.9+ 달성을 위한 Version 2 개선**")
    print("="*60)
    print("🚨 핵심 목표: 집중하는 학생을 '집중안함'으로 잘못 판단하는 것 방지")
    print("📈 전략: 보수적 예측, 높은 확신도에서만 '집중안함' 판단")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 저장 경로
    precision_dir = "./log/precision_v2"
    os.makedirs(precision_dir, exist_ok=True)
    os.makedirs(f"{precision_dir}/confusion_matrix", exist_ok=True)
    
    # 데이터 로드
    base_path = r"C:\Users\user\Desktop\brainbuddy_AI\preprocess2\pickle_labels"
    train_pkl_files = [
        f"{base_path}\\train\\20_01.pkl",
        f"{base_path}\\train\\20_03.pkl"
    ]
    val_pkl_files = [
        f"{base_path}\\valid\\20_01.pkl",
        f"{base_path}\\valid\\20_03.pkl"
    ]

    train_data_list = load_data(train_pkl_files)
    val_data_list = load_data(val_pkl_files)

    print(f"📊 훈련 데이터: {len(train_data_list):,}개")
    print(f"📊 검증 데이터: {len(val_data_list):,}개")

    # 데이터셋 생성
    train_dataset = PrecisionEnhancedDataset(train_data_list, is_training=True)
    val_dataset = PrecisionEnhancedDataset(val_data_list, is_training=False)

    # DataLoader
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"📊 배치 크기: {batch_size}")
    print(f"📊 훈련 배치: {len(train_loader):,}개")

    # 모델 초기화
    cnn = PrecisionEnhancedCNNEncoder().to(device)
    model = PrecisionEnhancedModelV2().to(device)
    
    print("✅ Precision Enhanced 모델 초기화 완료")
    print(f"   - CNN: 3층 FC + BatchNorm")
    print(f"   - Transformer: d_model=256, 4 layers")
    print(f"   - 특별 기능: Attention Pooling, Confidence Scaling")
    
    # Precision 최적화 손실 함수
    criterion = PrecisionFocusedLoss(precision_weight=2.5, pos_weight=1.3)
    
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(model.parameters()), 
        lr=1.5e-4,  # 더 낮은 학습률로 안정적 훈련
        weight_decay=2e-3  # 강한 정규화
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=5e-6)
    scaler = GradScaler()
    
    # 저장 경로
    best_model_path = f"{precision_dir}/best_precision_model.pt"
    checkpoint_path = f"{precision_dir}/last_precision_checkpoint.pt"
    log_history = []

    print(f"🎯 목표: Precision 0.9+ (집중하는 학생을 잘못 지적하지 않기)")
    print("="*60)

    # 훈련 루프
    num_epochs = 6
    patience = 3
    patience_counter = 0
    best_precision = 0.0

    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n[Epoch {epoch+1}/{num_epochs}] LR: {current_lr:.2e}")
        
        # 훈련
        start_time = time.time()
        train_loss = train_precision_model(model, cnn, train_loader, criterion, optimizer, device, scaler)
        train_time = time.time() - start_time
        
        # 검증
        start_time = time.time()
        val_loss = validate_precision_model(model, cnn, val_loader, criterion, device)
        val_time = time.time() - start_time
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"[Epoch {epoch+1}] Time - Train: {train_time/60:.1f}분, Val: {val_time/60:.1f}분")

        # 성능 평가 (높은 임계값으로)
        def create_sample_batches(loader, max_batches=50):
            """메모리 효율적인 샘플링"""
            sample_batches = []
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                sample_batches.append(batch)
                torch.cuda.empty_cache()  # 배치마다 메모리 정리
            return sample_batches

        # 사용
        sample_loader = create_sample_batches(val_loader, max_batches=50)
        accuracy, precision, recall, f1, all_probs = evaluate_precision_metrics(
            model, cnn, sample_loader, device, threshold=0.8
        )
        
        print(f"[Epoch {epoch+1}] Metrics (threshold=0.8)")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f} ⭐")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")

        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "learning_rate": current_lr,
            "train_time_min": train_time/60,
            "val_time_min": val_time/60
        })

        # 베스트 모델 저장 (Precision 기준)
        if precision > best_precision:
            best_precision = precision
            
            # 전체 검증 데이터로 임계값 최적화
            print("\n🔍 전체 검증 데이터로 최적 임계값 탐색 중...")
            all_sample_loader = list(val_loader)[:300]  # 더 많은 샘플로 정확한 측정
            _, _, _, _, all_validation_probs = evaluate_precision_metrics(
                model, cnn, all_sample_loader, device, threshold=0.5
            )
            
            all_validation_labels = []
            for videos, fusion, labels in all_sample_loader:
                all_validation_labels.extend(labels.int().numpy().flatten())
            
            optimal_threshold, best_metrics = find_optimal_threshold_for_precision(
                all_validation_probs, all_validation_labels, target_precision=0.9
            )
            
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'optimal_threshold': optimal_threshold,
                'best_metrics': best_metrics,
                'precision_focused': True
            }, best_model_path)
            
            print(f"✅ Best precision model saved (Precision: {precision:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Precision improvement patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("==== Early stopping Triggered (Precision 기준) ====")
                break

        # 체크포인트 저장
        torch.save({
            'cnn_state_dict': cnn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_precision': best_precision
        }, checkpoint_path)
        
        scheduler.step()

    # 로그 저장
    log_df = pd.DataFrame(log_history)
    log_df.to_csv(f"{precision_dir}/precision_training_log.csv", index=False)
    print(f"\n📄 Training log saved to {precision_dir}/precision_training_log.csv")

    # 최종 결과
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        final_precision = checkpoint.get('precision', 0)
        final_accuracy = checkpoint.get('accuracy', 0)
        final_recall = checkpoint.get('recall', 0)
        final_f1 = checkpoint.get('f1_score', 0)
        optimal_threshold = checkpoint.get('optimal_threshold', 0.8)
        best_metrics = checkpoint.get('best_metrics', {})
        
        print("\n" + "="*60)
        print("🎉 **Precision Enhanced Version 2 훈련 완료!**")
        print("="*60)
        print(f"🔸 기존 Version 2: 76.9% 정확도")
        print(f"🔸 Precision Enhanced: {final_accuracy:.1%} 정확도")
        print(f"🔸 핵심 성과 - Precision: {final_precision:.1%} ⭐")
        print(f"🔸 Recall: {final_recall:.1%}")
        print(f"🔸 F1-Score: {final_f1:.1%}")
        print(f"🔸 최적 임계값: {optimal_threshold:.2f}")
        
        if best_metrics:
            print(f"\n🏆 **Precision 0.9+ 달성 성공!**")
            print(f"   - 최종 Precision: {best_metrics['precision']:.3f}")
            print(f"   - 최종 Recall: {best_metrics['recall']:.3f}")
            print(f"   - 최종 F1-Score: {best_metrics['f1']:.3f}")
            print(f"   - 최종 Accuracy: {best_metrics['accuracy']:.3f}")
            print(f"📚 교육적 의미: 집중하는 학생을 잘못 지적할 확률 < 10%")
        
        print(f"📁 모델 저장: {best_model_path}")
        
        if final_precision >= 0.9:
            print("🎉 Precision 0.9+ 달성! 안전한 집중도 탐지 시스템 완성!")
        elif final_precision >= 0.85:
            print("🎊 Precision 0.85+ 달성! 실용적 수준의 신뢰성 확보!")
        else:
            print("📊 추가 개선 여지가 있지만, 기존 대비 향상 확인")
            
        print("\n🎯 **실제 활용 가이드**")
        print(f"   - 권장 임계값: {optimal_threshold:.2f}")
        print(f"   - AI가 '집중안함'이라고 할 때 신뢰도: {final_precision:.1%}")
        print(f"   - 실제 집중 안 하는 학생 탐지율: {final_recall:.1%}")

if __name__ == '__main__':
    main()
