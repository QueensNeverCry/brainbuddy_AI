import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureEmbeddingNetwork(nn.Module):
    """31차원 특징을 임베딩하는 네트워크"""
    
    def __init__(self, input_dim=31, embed_dim=128, dropout=0.1):
        super(FeatureEmbeddingNetwork, self).__init__()
        
        # 특징별 가중치 (26차원 기본 특징 + 5차원 attention 특징)
        self.basic_feature_weight = nn.Parameter(torch.ones(26))
        self.attention_feature_weight = nn.Parameter(torch.ones(5))
        
        # 임베딩 레이어들
        self.embedding_layers = nn.ModuleList([
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        ])
        
        # 정규화
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, 31] 또는 [batch_size, 31]
        Returns:
            embedded_features: [batch_size, sequence_length, embed_dim] 또는 [batch_size, embed_dim]
        """
        # 특징 가중치 적용
        if len(x.shape) == 3:  # 시퀀스 데이터
            batch_size, seq_len, _ = x.shape
            weights = torch.cat([self.basic_feature_weight, self.attention_feature_weight])
            x = x * weights.unsqueeze(0).unsqueeze(0)  # [1, 1, 31]
        else:  # 단일 프레임
            weights = torch.cat([self.basic_feature_weight, self.attention_feature_weight])
            x = x * weights.unsqueeze(0)  # [1, 31]
        
        # 순차적 임베딩
        for layer in self.embedding_layers:
            x = layer(x)
        
        # 정규화
        x = self.layer_norm(x)
        
        return x


class AdaptiveFeatureEncoder(nn.Module):
    """적응형 특징 인코더"""
    
    def __init__(self, input_dim=31, hidden_dim=256, output_dim=128):
        super(AdaptiveFeatureEncoder, self).__init__()
        
        # 특징 타입별 인코더
        self.pose_encoder = nn.Linear(3, hidden_dim // 4)      # [0:3] 머리 포즈
        self.gaze_encoder = nn.Linear(2, hidden_dim // 4)      # [4:6] 시선
        self.eye_encoder = nn.Linear(6, hidden_dim // 4)       # [6:12] 눈 관련
        self.stability_encoder = nn.Linear(16, hidden_dim // 4) # [13:29] 안정성 등
        self.attention_encoder = nn.Linear(5, hidden_dim // 4)  # [26:31] attention 특징
        
        # 융합 네트워크
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """특징별 인코딩 후 융합"""
        # 특징 분할
        pose_feat = x[..., 0:3]         # 머리 포즈
        gaze_feat = x[..., 4:6]         # 시선 (거리 제외)
        eye_feat = x[..., 6:12]         # 눈 관련
        stability_feat = x[..., 12:26]  # 안정성 관련
        attention_feat = x[..., 26:31]  # attention 특징
        
        # 각각 인코딩
        pose_encoded = self.pose_encoder(pose_feat)
        gaze_encoded = self.gaze_encoder(gaze_feat)
        eye_encoded = self.eye_encoder(eye_feat)
        stability_encoded = self.stability_encoder(stability_feat)
        attention_encoded = self.attention_encoder(attention_feat)
        
        # 특징 융합
        fused = torch.cat([
            pose_encoded, gaze_encoded, eye_encoded, 
            stability_encoded, attention_encoded
        ], dim=-1)
        
        # 최종 인코딩
        output = self.fusion_network(fused)
        
        return output


class TemporalFeatureExtractor(nn.Module):
    """시계열 특징 추출기"""
    
    def __init__(self, input_dim=31, hidden_dim=128, num_layers=2):
        super(TemporalFeatureExtractor, self).__init__()
        
        # 1D CNN으로 지역적 패턴 추출
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM으로 전역적 패턴 추출
        self.global_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 시간 가중치 (최근 프레임에 더 높은 가중치)
        self.temporal_weights = nn.Parameter(torch.linspace(0.1, 1.0, 30))
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, sequence_length, input_dim] = [B, 30, 31]
        Returns:
            temporal_features: [batch_size, hidden_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 1D CNN 적용 (차원 변환 필요)
        x_transposed = x.transpose(1, 2)  # [B, 31, 30]
        local_features = self.local_conv(x_transposed)  # [B, hidden_dim, 30]
        local_features = local_features.transpose(1, 2)  # [B, 30, hidden_dim]
        
        # LSTM 적용
        lstm_out, _ = self.global_lstm(local_features)  # [B, 30, hidden_dim]
        
        # 시간 가중 평균
        weights = self.temporal_weights.unsqueeze(0).unsqueeze(-1)  # [1, 30, 1]
        weighted_features = lstm_out * weights
        temporal_features = torch.sum(weighted_features, dim=1)  # [B, hidden_dim]
        
        return temporal_features


class MultiScaleFeatureExtractor(nn.Module):
    """다중 스케일 특징 추출기"""
    
    def __init__(self, input_dim=31, base_dim=64):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 다양한 커널 크기의 1D CNN
        self.conv_3 = nn.Conv1d(input_dim, base_dim, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(input_dim, base_dim, kernel_size=5, padding=2)
        self.conv_7 = nn.Conv1d(input_dim, base_dim, kernel_size=7, padding=3)
        
        # 글로벌 풀링
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 특징 융합
        self.fusion = nn.Sequential(
            nn.Linear(base_dim * 3, base_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """다중 스케일 특징 추출"""
        # x: [B, 30, 31] -> [B, 31, 30]
        x = x.transpose(1, 2)
        
        # 각기 다른 스케일 적용
        feat_3 = F.relu(self.conv_3(x))  # [B, base_dim, 30]
        feat_5 = F.relu(self.conv_5(x))  # [B, base_dim, 30]
        feat_7 = F.relu(self.conv_7(x))  # [B, base_dim, 30]
        
        # 글로벌 풀링
        feat_3 = self.global_pool(feat_3).squeeze(-1)  # [B, base_dim]
        feat_5 = self.global_pool(feat_5).squeeze(-1)  # [B, base_dim]
        feat_7 = self.global_pool(feat_7).squeeze(-1)  # [B, base_dim]
        
        # 융합
        fused = torch.cat([feat_3, feat_5, feat_7], dim=-1)  # [B, base_dim*3]
        output = self.fusion(fused)  # [B, base_dim]
        
        return output


class AttentionFeatureProcessor(nn.Module):
    """Attention 특징 전용 처리기"""
    
    def __init__(self, attention_dim=5, output_dim=32):
        super(AttentionFeatureProcessor, self).__init__()
        
        # attention 특징별 가중치
        self.attention_weights = nn.Parameter(torch.tensor([
            1.2,  # central_focus (중요)
            1.1,  # gaze_fixation (중요) 
            0.9,  # head_stability
            0.8,  # face_orientation
            1.3   # attention_score (가장 중요)
        ]))
        
        # 처리 네트워크
        self.processor = nn.Sequential(
            nn.Linear(attention_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()  # 0~1 범위 유지
        )
        
    def forward(self, attention_features):
        """
        Args:
            attention_features: [..., 5] attention 특징
        Returns:
            processed: [..., output_dim] 처리된 특징
        """
        # 가중치 적용
        weighted = attention_features * self.attention_weights
        
        # 처리
        processed = self.processor(weighted)
        
        return processed


# 통합 특징 네트워크
class ComprehensiveFeatureNetwork(nn.Module):
    """통합 특징 처리 네트워크"""
    
    def __init__(self, input_dim=31, output_dim=256):
        super(ComprehensiveFeatureNetwork, self).__init__()
        
        # 서브 네트워크들
        self.feature_embedding = FeatureEmbeddingNetwork(input_dim, 64)
        self.adaptive_encoder = AdaptiveFeatureEncoder(input_dim, 128, 64)
        self.attention_processor = AttentionFeatureProcessor(5, 32)
        self.multiscale_extractor = MultiScaleFeatureExtractor(input_dim, 64)
        
        # 최종 융합
        self.final_fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 64, output_dim),  # 총 224차원 입력
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """종합적 특징 처리"""
        # 각 서브네트워크 적용
        embedded = self.feature_embedding(x)  # [64]
        adaptive = self.adaptive_encoder(x)   # [64]
        
        # attention 특징만 분리 처리
        attention_feat = x[..., 26:31]
        attention_processed = self.attention_processor(attention_feat)  # [32]
        
        # 시퀀스가 있는 경우만 multiscale 적용
        if len(x.shape) == 3:  # [B, seq, dim]
            multiscale = self.multiscale_extractor(x)  # [64]
        else:
            # 단일 프레임인 경우 더미 텐서
            multiscale = torch.zeros(x.shape[0], 64, device=x.device)
        
        # 모든 특징 융합
        fused = torch.cat([embedded, adaptive, attention_processed, multiscale], dim=-1)
        output = self.final_fusion(fused)
        
        return output


def test_feature_networks():
    """특징 네트워크 테스트"""
    print("🧪 특징 네트워크 테스트")
    
    # 테스트 데이터 생성
    batch_size = 4
    seq_len = 30
    input_dim = 31
    
    # 시퀀스 데이터
    sequence_data = torch.randn(batch_size, seq_len, input_dim)
    
    # 단일 프레임 데이터
    single_frame = torch.randn(batch_size, input_dim)
    
    # 1. 특징 임베딩 테스트
    embedding_net = FeatureEmbeddingNetwork()
    embedded_seq = embedding_net(sequence_data)
    embedded_single = embedding_net(single_frame)
    
    print(f"✅ 특징 임베딩: {sequence_data.shape} -> {embedded_seq.shape}")
    print(f"✅ 단일 임베딩: {single_frame.shape} -> {embedded_single.shape}")
    
    # 2. 적응형 인코더 테스트
    adaptive_net = AdaptiveFeatureEncoder()
    adaptive_out = adaptive_net(sequence_data)
    print(f"✅ 적응형 인코더: {sequence_data.shape} -> {adaptive_out.shape}")
    
    # 3. 시계열 추출기 테스트
    temporal_net = TemporalFeatureExtractor()
    temporal_out = temporal_net(sequence_data)
    print(f"✅ 시계열 추출: {sequence_data.shape} -> {temporal_out.shape}")
    
    # 4. 통합 네트워크 테스트
    comprehensive_net = ComprehensiveFeatureNetwork()
    comprehensive_out = comprehensive_net(sequence_data)
    print(f"✅ 통합 네트워크: {sequence_data.shape} -> {comprehensive_out.shape}")
    
    print("🎉 모든 특징 네트워크 테스트 완료!")

if __name__ == "__main__":
    test_feature_networks()
