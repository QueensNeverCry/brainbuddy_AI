import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse

from models.pytorch_concentration import create_model

class ConcentrationTrainer:
    """집중도 모델 학습기"""
    
    def __init__(self, model_type='lstm', lr=0.001, batch_size=32, epochs=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        print(f"🎯 학습 설정")
        print(f"  모델: {model_type}")
        print(f"  디바이스: {self.device}")
        print(f"  학습률: {lr}")
        print(f"  배치 크기: {batch_size}")
        print(f"  에폭: {epochs}")
        
    def load_dataset(self, dataset_path):
        """데이터셋 로드"""
        print(f"📂 데이터셋 로드: {dataset_path}")
        
        dataset = torch.load(dataset_path)
        sequences = dataset['sequences']
        labels = dataset['labels']
        
        print(f"  시퀀스 수: {len(sequences)}")
        print(f"  특징 차원: {dataset['feature_dim']}")
        print(f"  시퀀스 길이: {dataset['sequence_length']}")
        
        # 텐서로 변환
        X = torch.stack(sequences)  # [N, 30, 31]
        y = torch.FloatTensor(labels).unsqueeze(1)  # [N, 1]
        
        # 데이터셋 분할 (8:1:1)
        total_size = len(sequences)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # 랜덤 분할
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 분할된 데이터
        train_X, train_y = X[train_indices], y[train_indices]
        val_X, val_y = X[val_indices], y[val_indices]
        test_X, test_y = X[test_indices], y[test_indices]
        
        # 데이터로더 생성
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"  학습 데이터: {len(train_dataset)}")
        print(f"  검증 데이터: {len(val_dataset)}")
        print(f"  테스트 데이터: {len(test_dataset)}")
        
        # 클래스 분포 확인
        train_focus_ratio = train_y.mean().item() * 100
        print(f"  학습 집중 비율: {train_focus_ratio:.1f}%")
        
        return dataset['feature_dim']
    
    def create_model(self, input_dim):
        """모델 생성"""
        self.model = create_model(self.model_type, input_dim=input_dim)
        self.model = self.model.to(self.device)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"🧠 모델 파라미터: {total_params:,}개")
        
        # 손실함수 및 옵티마이저
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
    def train_epoch(self):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """한 에폭 검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(self.val_loader, desc="Validation"):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # 예측 수집
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels = batch_y.cpu().numpy()
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())
        
        # 메트릭 계산
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc': roc_auc_score(all_labels, all_probs)
        }
        
        return metrics
    
    def train(self):
        """전체 학습 과정"""
        print(f"\n🚀 학습 시작")
        
        train_losses = []
        val_metrics = []
        best_f1 = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 학습
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 검증
            val_metrics_epoch = self.validate_epoch()
            val_metrics.append(val_metrics_epoch)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 로깅
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics_epoch['loss']:.4f}")
            print(f"  Val Acc: {val_metrics_epoch['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics_epoch['f1']:.4f}")
            print(f"  Val AUC: {val_metrics_epoch['auc']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 모델 저장
            current_f1 = val_metrics_epoch['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                
                # 최고 성능 모델 저장
                os.makedirs('./checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': current_f1,
                    'val_metrics': val_metrics_epoch,
                    'model_type': self.model_type
                }, f'./checkpoints/best_{self.model_type}_concentration.pt')
                
                print(f"  ✅ 최고 성능 모델 저장 (F1: {current_f1:.4f})")
            else:
                patience_counter += 1
            
            # Early Stopping
            if patience_counter >= patience:
                print(f"  ⏹️ Early stopping (patience: {patience})")
                break
        
        # 학습 곡선 저장
        self.plot_training_curves(train_losses, val_metrics)
        
        print(f"\n✅ 학습 완료!")
        print(f"최고 F1 점수: {best_f1:.4f}")
        
        return best_f1
    
    def plot_training_curves(self, train_losses, val_metrics):
        """학습 곡선 시각화"""
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
        val_losses = [m['loss'] for m in val_metrics]
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        val_accs = [m['accuracy'] for m in val_metrics]
        axes[0, 1].plot(epochs, val_accs, 'g-', label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        val_f1s = [m['f1'] for m in val_metrics]
        axes[1, 0].plot(epochs, val_f1s, 'm-', label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC
        val_aucs = [m['auc'] for m in val_metrics]
        axes[1, 1].plot(epochs, val_aucs, 'c-', label='Val AUC')
        axes[1, 1].set_title('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./checkpoints/{self.model_type}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 학습 곡선 저장: ./checkpoints/{self.model_type}_training_curves.png")

def main():
    parser = argparse.ArgumentParser(description='집중도 PyTorch 모델 학습')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'transformer', 'cnn1d'],
                       help='모델 타입')
    parser.add_argument('--data', type=str, default='./data/concentration_sequences.pt',
                       help='데이터셋 경로')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    
    args = parser.parse_args()
    
    print("🧠 PyTorch 집중도 모델 학습")
    print(f"모델: {args.model}")
    
    # 데이터셋 확인
    if not os.path.exists(args.data):
        print(f"❌ 데이터셋을 찾을 수 없습니다: {args.data}")
        print("먼저 convert_ml_to_pytorch.py를 실행하여 데이터를 생성하세요.")
        return
    
    # 학습기 생성
    trainer = ConcentrationTrainer(
        model_type=args.model,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # 데이터 로드
    input_dim = trainer.load_dataset(args.data)
    
    # 모델 생성
    trainer.create_model(input_dim)
    
    # 학습 실행
    best_f1 = trainer.train()
    
    print(f"\n🎉 학습 완료!")
    print(f"최고 성능 모델: ./checkpoints/best_{args.model}_concentration.pt")
    print(f"최고 F1 점수: {best_f1:.4f}")

if __name__ == "__main__":
    main()
