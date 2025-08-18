import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ConcentrationModelTrainer:
    """집중도 모델 전문 학습기"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model.to(self.device)
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.learning_rates = []
        
        # Early stopping
        self.best_val_score = 0.0
        self.patience_counter = 0
        
        print(f"🎯 학습기 초기화 완료 (디바이스: {self.device})")
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-4, 
                      scheduler_type: str = 'cosine'):
        """학습 설정"""
        # 손실함수
        self.criterion = nn.BCELoss()
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 스케줄러
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            self.scheduler = None
        
        print(f"⚙️ 학습 설정 완료 (LR: {learning_rate}, Scheduler: {scheduler_type})")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (sequences, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (선택적)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 손실 기록
            batch_loss = loss.item()
            total_loss += batch_loss
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """한 에폭 검증"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for sequences, labels in progress_bar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # 예측 결과 수집
                probabilities = outputs.cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                
                all_probabilities.extend(probabilities.flatten())
                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels_np.flatten())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 메트릭 계산
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probabilities) if len(set(all_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 15, save_dir: str = './checkpoints') -> Dict:
        """전체 학습 과정"""
        
        print(f"🚀 학습 시작 (에폭: {epochs}, 인내심: {patience})")
        print(f"📊 학습 데이터: {len(train_loader.dataset)}개")
        print(f"📊 검증 데이터: {len(val_loader.dataset)}개")
        
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 검증
            val_metrics = self.validate_epoch(val_loader, epoch)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            # 학습률 기록
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # 에폭 시간 계산
            epoch_time = time.time() - epoch_start
            
            # 로그 출력
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"{'='*60}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"Val Prec:   {val_metrics['precision']:.4f}")
            print(f"Val Rec:    {val_metrics['recall']:.4f}")
            print(f"Val F1:     {val_metrics['f1']:.4f}")
            print(f"Val AUC:    {val_metrics['auc']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # 모델 저장 (F1 score 기준)
            current_score = val_metrics['f1']
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.patience_counter = 0
                
                # 최고 성능 모델 저장
                model_name = self.model.__class__.__name__.lower()
                save_path = os.path.join(save_dir, f'best_{model_name}_concentration.pt')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_f1': current_score,
                    'val_metrics': val_metrics,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_metrics': self.val_metrics,
                    'model_type': model_name
                }, save_path)
                
                print(f"✅ 최고 성능 모델 저장: {save_path}")
                print(f"🏆 최고 F1 점수: {current_score:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ Early stopping 카운터: {self.patience_counter}/{patience}")
            
            # Early stopping 확인
            if self.patience_counter >= patience:
                print(f"⏹️ Early stopping 발동 (patience: {patience})")
                break
            
            print(f"{'='*60}\n")
        
        # 학습 완료
        total_time = time.time() - start_time
        print(f"🎉 학습 완료! (총 시간: {total_time/60:.1f}분)")
        print(f"🏆 최고 F1 점수: {self.best_val_score:.4f}")
        
        return {
            'best_f1': self.best_val_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'total_time': total_time
        }
    
    def plot_training_history(self, save_path: str = None):
        """학습 과정 시각화"""
        if not self.val_metrics:
            print("❌ 학습 기록이 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss 곡선
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        val_accuracies = [m['accuracy'] for m in self.val_metrics]
        axes[0, 1].plot(epochs, val_accuracies, 'g-', linewidth=2)
        axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        val_f1s = [m['f1'] for m in self.val_metrics]
        axes[0, 2].plot(epochs, val_f1s, 'm-', linewidth=2)
        axes[0, 2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision & Recall
        val_precisions = [m['precision'] for m in self.val_metrics]
        val_recalls = [m['recall'] for m in self.val_metrics]
        axes[1, 0].plot(epochs, val_precisions, 'c-', label='Precision', linewidth=2)
        axes[1, 0].plot(epochs, val_recalls, 'y-', label='Recall', linewidth=2)
        axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        val_aucs = [m['auc'] for m in self.val_metrics]
        axes[1, 1].plot(epochs, val_aucs, 'orange', linewidth=2)
        axes[1, 1].set_title('Validation AUC', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 2].plot(epochs, self.learning_rates, 'purple', linewidth=2)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 학습 곡선 저장: {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_loader: DataLoader, save_dir: str = './results'):
        """모델 최종 평가"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        print("🔍 최종 모델 평가 중...")
        
        with torch.no_grad():
            for sequences, labels in tqdm(test_loader, desc="Testing"):
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                
                probabilities = outputs.cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                
                all_probabilities.extend(probabilities.flatten())
                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels_np.flatten())
        
        # 평가 메트릭
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        test_metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probabilities)
        }
        
        # 결과 출력
        print(f"\n{'='*50}")
        print(f"📊 최종 테스트 결과")
        print(f"{'='*50}")
        for metric, value in test_metrics.items():
            print(f"{metric.upper():>10}: {value:.4f}")
        print(f"{'='*50}")
        
        # Confusion Matrix 시각화
        os.makedirs(save_dir, exist_ok=True)
        self._plot_confusion_matrix(all_labels, all_predictions, 
                                   os.path.join(save_dir, 'confusion_matrix.png'))
        
        # Classification Report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=['Not Focused', 'Focused'])
        print(f"\n📋 분류 리포트:\n{report}")
        
        return test_metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_path: str):
        """혼동 행렬 시각화"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Focused', 'Focused'],
                   yticklabels=['Not Focused', 'Focused'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 혼동 행렬 저장: {save_path}")
