import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcentrationLSTM(nn.Module):
    """30프레임 시퀀스 기반 집중도 분석 LSTM 모델"""
    
    def __init__(self, input_dim=31, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ConcentrationLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 입력 정규화
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 특징 임베딩
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim] = [B, 30, 31]
        Returns:
            output: [batch_size, 1] 집중 확률
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 정규화
        x = self.input_norm(x)
        
        # 특징 임베딩
        embedded = self.feature_embedding(x)  # [B, 30, hidden_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [B, 30, hidden_dim*2]
        
        # Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, 30, hidden_dim*2]
        
        # 시퀀스 풀링 (마지막 10프레임 평균)
        # 최근 프레임이 더 중요하므로 가중 평균
        weights = torch.softmax(torch.arange(seq_len, dtype=torch.float, device=x.device), dim=0)
        pooled = torch.sum(attn_out * weights.view(1, -1, 1), dim=1)  # [B, hidden_dim*2]
        
        # 분류
        output = self.classifier(pooled)  # [B, 1]
        
        return output


class ConcentrationTransformer(nn.Module):
    """Transformer 기반 집중도 분석 모델"""
    
    def __init__(self, input_dim=31, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(ConcentrationTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoding = nn.Parameter(torch.randn(30, d_model))
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            output: [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # 입력 프로젝션
        x = self.input_projection(x)  # [B, 30, d_model]
        
        # 위치 인코딩 추가
        x = x + self.pos_encoding.unsqueeze(0)  # [B, 30, d_model]
        
        # Transformer
        transformer_out = self.transformer(x)  # [B, 30, d_model]
        
        # Global Average Pooling
        pooled = transformer_out.mean(dim=1)  # [B, d_model]
        
        # 분류
        output = self.classifier(pooled)  # [B, 1]
        
        return output


class ConcentrationCNN1D(nn.Module):
    """1D CNN 기반 경량 모델"""
    
    def __init__(self, input_dim=31, num_filters=64, dropout=0.3):
        super(ConcentrationCNN1D, self).__init__()
        
        # 1D Convolution layers
        self.conv_layers = nn.Sequential(
            # 첫 번째 블록
            nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 두 번째 블록
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # 세 번째 블록
            nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        
        # 전역 평균 풀링
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 4, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim]
        Returns:
            output: [batch_size, 1]
        """
        # Transpose for 1D conv: [B, input_dim, sequence_length]
        x = x.transpose(1, 2)  # [B, 31, 30]
        
        # 1D Convolution
        conv_out = self.conv_layers(x)  # [B, num_filters*4, reduced_length]
        
        # Global pooling
        pooled = self.global_pool(conv_out).squeeze(-1)  # [B, num_filters*4]
        
        # 분류
        output = self.classifier(pooled)  # [B, 1]
        
        return output


def create_model(model_type='lstm', **kwargs):
    """모델 팩토리 함수"""
    if model_type == 'lstm':
        return ConcentrationLSTM(**kwargs)
    elif model_type == 'transformer':
        return ConcentrationTransformer(**kwargs)
    elif model_type == 'cnn1d':
        return ConcentrationCNN1D(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
