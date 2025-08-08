import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TrainingConfig:
    """학습 설정 클래스"""
    
    # 모델 설정
    model_type: str = 'lstm'  # lstm, transformer, cnn1d
    model_params: Dict[str, Any] = None
    
    # 학습 설정
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    scheduler_type: str = 'cosine'  # cosine, step, plateau
    
    # 데이터 설정
    dataset_path: str = './data/concentration_sequences.pt'
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    num_workers: int = 4
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # 저장 설정
    save_dir: str = './checkpoints'
    save_every: int = 10
    
    # 하드웨어 설정
    device: str = 'auto'  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # 로깅 설정
    log_interval: int = 10
    plot_training: bool = True
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = self.get_default_model_params()
    
    def get_default_model_params(self) -> Dict[str, Any]:
        """모델별 기본 파라미터"""
        if self.model_type == 'lstm':
            return {
                'input_dim': 31,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.3
            }
        elif self.model_type == 'transformer':
            return {
                'input_dim': 31,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1
            }
        elif self.model_type == 'cnn1d':
            return {
                'input_dim': 31,
                'num_filters': 64,
                'dropout': 0.3
            }
        else:
            return {'input_dim': 31}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'scheduler_type': self.scheduler_type,
            'dataset_path': self.dataset_path,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'patience': self.patience,
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """딕셔너리에서 생성"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """설정 저장"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 설정 저장: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """설정 로드"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# 사전 정의된 설정들
PRESET_CONFIGS = {
    'lstm_basic': TrainingConfig(
        model_type='lstm',
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        scheduler_type='cosine'
    ),
    
    'lstm_advanced': TrainingConfig(
        model_type='lstm',
        model_params={
            'input_dim': 31,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.2
        },
        learning_rate=0.0005,
        batch_size=64,
        epochs=100,
        scheduler_type='cosine',
        patience=20
    ),
    
    'transformer_basic': TrainingConfig(
        model_type='transformer',
        learning_rate=0.0001,
        batch_size=16,
        epochs=80,
        scheduler_type='cosine',
        patience=15
    ),
    
    'cnn1d_fast': TrainingConfig(
        model_type='cnn1d',
        learning_rate=0.002,
        batch_size=64,
        epochs=30,
        scheduler_type='step',
        patience=10
    )
}

def get_preset_config(preset_name: str) -> TrainingConfig:
    """사전 정의된 설정 가져오기"""
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"알 수 없는 preset: {preset_name}. 사용 가능: {available}")
    
    return PRESET_CONFIGS[preset_name]


# 하이퍼파라미터 튜닝용 설정
class HyperparameterSearch:
    """하이퍼파라미터 검색 공간 정의"""
    
    LSTM_SEARCH_SPACE = {
        'learning_rate': [0.001, 0.0005, 0.002],
        'hidden_dim': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.3, 0.5],
        'batch_size': [16, 32, 64]
    }
    
    TRANSFORMER_SEARCH_SPACE = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'd_model': [128, 256, 512],
        'nhead': [4, 8, 16],
        'num_layers': [3, 6, 9],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [8, 16, 32]
    }
    
    CNN1D_SEARCH_SPACE = {
        'learning_rate': [0.001, 0.002, 0.005],
        'num_filters': [32, 64, 128],
        'dropout': [0.2, 0.3, 0.4],
        'batch_size': [32, 64, 128]
    }
    
    @classmethod
    def get_search_space(cls, model_type: str) -> Dict[str, list]:
        """모델별 검색 공간 반환"""
        if model_type == 'lstm':
            return cls.LSTM_SEARCH_SPACE
        elif model_type == 'transformer':
            return cls.TRANSFORMER_SEARCH_SPACE
        elif model_type == 'cnn1d':
            return cls.CNN1D_SEARCH_SPACE
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
