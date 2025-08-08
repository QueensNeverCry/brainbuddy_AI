from flask import Flask, request, jsonify, render_template_string
import torch
import numpy as np
import base64
import cv2
import os
import time
from typing import Dict, List, Optional
import threading
import queue
from collections import deque

from models.pytorch_concentration import create_model
from utils.face_detector import FaceDetector
from utils.attention_features import AttentionFeatureExtractor

app = Flask(__name__)

class ConcentrationAPI:
    """백엔드 연동용 집중도 분석 API"""
    
    def __init__(self, model_path: str, model_type: str = 'lstm'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # 모델 로드
        self.load_model(model_path)
        
        # 유틸리티
        self.face_detector = FaceDetector()
        self.feature_extractor = AttentionFeatureExtractor()
        
        # 30프레임 버퍼 (클라이언트별)
        self.client_buffers = {}
        self.analysis_results = deque(maxlen=1000)  # 최근 1000개 결과 보관
        
        # 통계
        self.total_requests = 0
        self.successful_analyses = 0
        
        print(f"✅ 집중도 분석 API 초기화 완료")
        print(f"모델: {model_type}")
        print(f"디바이스: {self.device}")
    
    def load_model(self, model_path: str):
        """PyTorch 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 생성
        self.model = create_model(self.model_type, input_dim=31)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.model_info = {
            'model_type': self.model_type,
            'val_f1': checkpoint.get('val_f1', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A')
        }
        
        print(f"📂 모델 로드 완료 (F1: {self.model_info['val_f1']:.4f})")
    
    def decode_image(self, image_b64: str) -> Optional[np.ndarray]:
        """Base64 이미지 디코딩"""
        try:
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"❌ 이미지 디코딩 오류: {e}")
            return None
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """프레임에서 특징 추출 (31차원)"""
        face_box = self.face_detector.detect_face(frame)
        features, attention_features = self.feature_extractor.extract_features(frame, face_box)
        
        # 31차원 결합 특징
        combined_features = np.concatenate([
            features,  # 26차원
            [
                attention_features['central_focus'],
                attention_features['gaze_fixation'],
                attention_features['head_stability'],
                attention_features['face_orientation'],
                attention_features['attention_score']
            ]  # 5차원
        ])
        
        return combined_features
    
    def predict_concentration(self, sequence: np.ndarray) -> tuple[int, float]:
        """30프레임 시퀀스로 집중도 예측"""
        # 텐서로 변환
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # [1, 30, 31]
        sequence_tensor = sequence_tensor.to(self.device)
        
        # 예측
        with torch.no_grad():
            output = self.model(sequence_tensor)
            confidence = output.item()
            prediction = 1 if confidence > 0.5 else 0
        
        return prediction, confidence
    
    def add_frame_to_buffer(self, client_id: str, frame_features: np.ndarray):
        """클라이언트별 프레임 버퍼에 추가"""
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = {
                'buffer': deque(maxlen=30),
                'last_update': time.time()
            }
        
        self.client_buffers[client_id]['buffer'].append(frame_features)
        self.client_buffers[client_id]['last_update'] = time.time()
    
    def can_analyze(self, client_id: str) -> bool:
        """30프레임이 준비되었는지 확인"""
        if client_id not in self.client_buffers:
            return False
        
        return len(self.client_buffers[client_id]['buffer']) == 30
    
    def cleanup_old_buffers(self, timeout: int = 300):
        """오래된 클라이언트 버퍼 정리 (5분)"""
        current_time = time.time()
        expired_clients = []
        
        for client_id, data in self.client_buffers.items():
            if current_time - data['last_update'] > timeout:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.client_buffers[client_id]
            print(f"🧹 만료된 클라이언트 버퍼 정리: {client_id}")

# 글로벌 API 인스턴스
concentration_api: Optional[ConcentrationAPI] = None

def init_api(model_path: str, model_type: str = 'lstm'):
    """API 초기화"""
    global concentration_api
    concentration_api = ConcentrationAPI(model_path, model_type)


# API 엔드포인트들
@app.route('/')
def home():
    """홈페이지"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧠 PyTorch 집중도 분석 API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .endpoint { background-color: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { color: #007bff; font-weight: bold; font-size: 14px; }
            .url { color: #28a745; font-family: monospace; font-size: 16px; }
            .description { margin: 10px 0; color: #666; }
            .code-block { background-color: #282c34; color: #abb2bf; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 12px; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat-item { text-align: center; padding: 15px; background-color: #e9ecef; border-radius: 5px; }
            .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 14px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 PyTorch 집중도 분석 API</h1>
                <p>30프레임 시퀀스 기반 실시간 집중도 분석 시스템</p>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">{{ api_stats.model_type.upper() }}</div>
                    <div class="stat-label">모델 타입</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ api_stats.device }}</div>
                    <div class="stat-label">디바이스</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ api_stats.total_requests }}</div>
                    <div class="stat-label">총 요청 수</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.2f"|format(api_stats.success_rate) }}%</div>
                    <div class="stat-label">성공률</div>
                </div>
            </div>

            <h2>📡 API 엔드포인트</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> <span class="url">/api/analyze_single</span></h3>
                <div class="description">단일 프레임 분석 (버퍼에 추가)</div>
                <p><strong>입력:</strong> JSON { "client_id": "string", "image": "base64_encoded_image" }</p>
                <p><strong>출력:</strong> JSON { "frames_collected": int, "ready_for_analysis": bool }</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> <span class="url">/api/analyze_sequence</span></h3>
                <div class="description">30프레임 시퀀스 분석</div>
                <p><strong>입력:</strong> JSON { "client_id": "string" }</p>
                <p><strong>출력:</strong> JSON { "result": 0|1, "confidence": float, "message": "string" }</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> <span class="url">/api/analyze_batch</span></h3>
                <div class="description">30개 이미지 배치 분석</div>
                <p><strong>입력:</strong> JSON { "images": ["base64_1", "base64_2", ..., "base64_30"] }</p>
                <p><strong>출력:</strong> JSON { "result": 0|1, "confidence": float }</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> <span class="url">/api/status</span></h3>
                <div class="description">서버 상태 확인</div>
                <p><strong>출력:</strong> JSON { "status": "healthy", "model_info": {...}, "statistics": {...} }</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> <span class="url">/api/statistics</span></h3>
                <div class="description">분석 통계 조회</div>
                <p><strong>출력:</strong> JSON { "total_analyses": int, "recent_results": [...] }</p>
            </div>

            <h2>💻 사용 예시</h2>
            <div class="code-block">
import requests
import base64
import json

# 1. 단일 프레임씩 전송
client_id = "user_123"
for i in range(30):
    with open(f'frame_{i}.jpg', 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    response = requests.post('http://localhost:5000/api/analyze_single', 
                            json={'client_id': client_id, 'image': img_b64})
    print(f"Frame {i+1}: {response.json()}")

# 2. 30프레임이 모이면 분석
response = requests.post('http://localhost:5000/api/analyze_sequence',
                        json={'client_id': client_id})
result = response.json()
print(f"집중도: {result['result']}, 확신도: {result['confidence']}")

# 3. 배치 분석 (30개 이미지 한번에)
images = []
for i in range(30):
    with open(f'frame_{i}.jpg', 'rb') as f:
        images.append(base64.b64encode(f.read()).decode('utf-8'))

response = requests.post('http://localhost:5000/api/analyze_batch',
                        json={'images': images})
result = response.json()
print(f"배치 분석 결과: {result}")
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
                <h4 style="color: #155724; margin: 0 0 10px 0;">🚀 시작하기</h4>
                <p style="margin: 0; color: #155724;">1. 웹캠에서 30프레임을 수집하세요</p>
                <p style="margin: 0; color: #155724;">2. API로 프레임들을 전송하세요</p>
                <p style="margin: 0; color: #155724;">3. 집중도 분석 결과를 받으세요 (0: 비집중, 1: 집중)</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # API 통계
    api_stats = {
        'model_type': concentration_api.model_type if concentration_api else 'None',
        'device': str(concentration_api.device) if concentration_api else 'None',
        'total_requests': concentration_api.total_requests if concentration_api else 0,
        'success_rate': (concentration_api.successful_analyses / max(concentration_api.total_requests, 1) * 100) if concentration_api else 0
    }
    
    return render_template_string(html_template, api_stats=api_stats)


@app.route('/api/analyze_single', methods=['POST'])
def analyze_single_frame():
    """단일 프레임 분석 (버퍼에 추가)"""
    if not concentration_api:
        return jsonify({'error': 'API가 초기화되지 않았습니다'}), 500
    
    concentration_api.total_requests += 1
    
    try:
        data = request.get_json()
        if not data or 'client_id' not in data or 'image' not in data:
            return jsonify({'error': 'client_id와 image가 필요합니다'}), 400
        
        client_id = data['client_id']
        image_b64 = data['image']
        
        # 이미지 디코딩
        frame = concentration_api.decode_image(image_b64)
        if frame is None:
            return jsonify({'error': '유효하지 않은 이미지'}), 400
        
        # 특징 추출 및 버퍼에 추가
        frame_features = concentration_api.extract_frame_features(frame)
        concentration_api.add_frame_to_buffer(client_id, frame_features)
        
        # 버퍼 상태 확인
        frames_collected = len(concentration_api.client_buffers[client_id]['buffer'])
        ready_for_analysis = concentration_api.can_analyze(client_id)
        
        concentration_api.successful_analyses += 1
        
        return jsonify({
            'success': True,
            'frames_collected': frames_collected,
            'ready_for_analysis': ready_for_analysis,
            'message': f'프레임 {frames_collected}/30 수집 완료'
        })
        
    except Exception as e:
        return jsonify({'error': f'프레임 분석 실패: {str(e)}'}), 500


@app.route('/api/analyze_sequence', methods=['POST'])
def analyze_sequence():
    """30프레임 시퀀스 분석"""
    if not concentration_api:
        return jsonify({'error': 'API가 초기화되지 않았습니다'}), 500
    
    concentration_api.total_requests += 1
    
    try:
        data = request.get_json()
        if not data or 'client_id' not in data:
            return jsonify({'error': 'client_id가 필요합니다'}), 400
        
        client_id = data['client_id']
        
        # 30프레임 준비 확인
        if not concentration_api.can_analyze(client_id):
            current_frames = len(concentration_api.client_buffers.get(client_id, {}).get('buffer', []))
            return jsonify({
                'error': f'30프레임이 필요합니다 (현재: {current_frames}프레임)',
                'frames_needed': 30 - current_frames
            }), 400
        
        # 시퀀스 분석
        sequence = np.array(list(concentration_api.client_buffers[client_id]['buffer']))
        prediction, confidence = concentration_api.predict_concentration(sequence)
        
        # 결과 저장
        result_data = {
            'client_id': client_id,
            'timestamp': time.time(),
            'result': prediction,
            'confidence': confidence
        }
        concentration_api.analysis_results.append(result_data)
        
        # 클라이언트 버퍼 초기화
        concentration_api.client_buffers[client_id]['buffer'].clear()
        
        concentration_api.successful_analyses += 1
        
        return jsonify({
            'result': prediction,
            'confidence': confidence,
            'message': f"{'집중' if prediction == 1 else '비집중'} 상태로 판정되었습니다",
            'timestamp': result_data['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': f'시퀀스 분석 실패: {str(e)}'}), 500


@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """30개 이미지 배치 분석"""
    if not concentration_api:
        return jsonify({'error': 'API가 초기화되지 않았습니다'}), 500
    
    concentration_api.total_requests += 1
    
    try:
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({'error': 'images 배열이 필요합니다'}), 400
        
        images = data['images']
        if len(images) != 30:
            return jsonify({'error': '정확히 30개 이미지가 필요합니다'}), 400
        
        # 각 이미지에서 특징 추출
        sequence_features = []
        for i, image_b64 in enumerate(images):
            frame = concentration_api.decode_image(image_b64)
            if frame is None:
                return jsonify({'error': f'이미지 {i+1}이 유효하지 않습니다'}), 400
            
            features = concentration_api.extract_frame_features(frame)
            sequence_features.append(features)
        
        # 시퀀스 분석
        sequence = np.array(sequence_features)
        prediction, confidence = concentration_api.predict_concentration(sequence)
        
        # 결과 저장
        result_data = {
            'client_id': 'batch_client',
            'timestamp': time.time(),
            'result': prediction,
            'confidence': confidence,
            'type': 'batch'
        }
        concentration_api.analysis_results.append(result_data)
        
        concentration_api.successful_analyses += 1
        
        return jsonify({
            'result': prediction,
            'confidence': confidence,
            'message': f"배치 분석 완료: {'집중' if prediction == 1 else '비집중'}",
            'processed_frames': 30
        })
        
    except Exception as e:
        return jsonify({'error': f'배치 분석 실패: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """서버 상태 확인"""
    if not concentration_api:
        return jsonify({'status': 'error', 'message': 'API가 초기화되지 않았습니다'}), 500
    
    # 오래된 버퍼 정리
    concentration_api.cleanup_old_buffers()
    
    return jsonify({
        'status': 'healthy',
        'model_info': concentration_api.model_info,
        'statistics': {
            'total_requests': concentration_api.total_requests,
            'successful_analyses': concentration_api.successful_analyses,
            'success_rate': concentration_api.successful_analyses / max(concentration_api.total_requests, 1),
            'active_clients': len(concentration_api.client_buffers),
            'recent_results_count': len(concentration_api.analysis_results)
        },
        'server_info': {
            'device': str(concentration_api.device),
            'model_type': concentration_api.model_type
        }
    })


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """분석 통계 조회"""
    if not concentration_api:
        return jsonify({'error': 'API가 초기화되지 않았습니다'}), 500
    
    # 최근 결과들
    recent_results = list(concentration_api.analysis_results)[-50:]  # 최근 50개
    
    if recent_results:
        focus_count = sum(1 for r in recent_results if r['result'] == 1)
        focus_ratio = focus_count / len(recent_results)
        
        avg_confidence = np.mean([r['confidence'] for r in recent_results])
        
        # 시간대별 분석
        current_time = time.time()
        recent_hour = [r for r in recent_results if current_time - r['timestamp'] < 3600]
        hour_focus_ratio = sum(1 for r in recent_hour if r['result'] == 1) / max(len(recent_hour), 1)
    else:
        focus_count = 0
        focus_ratio = 0.0
        avg_confidence = 0.0
        hour_focus_ratio = 0.0
    
    return jsonify({
        'total_analyses': len(concentration_api.analysis_results),
        'recent_results': {
            'count': len(recent_results),
            'focus_count': focus_count,
            'focus_ratio': focus_ratio,
            'avg_confidence': avg_confidence
        },
        'hourly_stats': {
            'analyses_last_hour': len(recent_hour) if 'recent_hour' in locals() else 0,
            'focus_ratio_last_hour': hour_focus_ratio
        },
        'latest_results': recent_results[-10:] if recent_results else []
    })


@app.route('/api/clear_client', methods=['POST'])
def clear_client_buffer():
    """클라이언트 버퍼 초기화"""
    if not concentration_api:
        return jsonify({'error': 'API가 초기화되지 않았습니다'}), 500
    
    try:
        data = request.get_json()
        if not data or 'client_id' not in data:
            return jsonify({'error': 'client_id가 필요합니다'}), 400
        
        client_id = data['client_id']
        
        if client_id in concentration_api.client_buffers:
            del concentration_api.client_buffers[client_id]
            return jsonify({
                'success': True,
                'message': f'클라이언트 {client_id}의 버퍼가 초기화되었습니다'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'클라이언트 {client_id}를 찾을 수 없습니다'
            })
    
    except Exception as e:
        return jsonify({'error': f'버퍼 초기화 실패: {str(e)}'}), 500


# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API 엔드포인트를 찾을 수 없습니다', 'available_endpoints': [
        'POST /api/analyze_single',
        'POST /api/analyze_sequence', 
        'POST /api/analyze_batch',
        'GET /api/status',
        'GET /api/statistics',
        'POST /api/clear_client'
    ]}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '서버 내부 오류가 발생했습니다'}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch 집중도 분석 API 서버')
    parser.add_argument('--model', type=str, required=True, help='PyTorch 모델 파일 경로')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'cnn1d'], help='모델 타입')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트')
    parser.add_argument('--port', type=int, default=5000, help='서버 포트')
    parser.add_argument('--debug', action='store_true', help='디버그 모드')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        exit(1)
    
    # API 초기화
    print("🚀 PyTorch 집중도 분석 API 서버 시작")
    print(f"모델: {args.model}")
    print(f"서버: http://{args.host}:{args.port}")
    
    init_api(args.model, args.model_type)
    
    # 서버 실행
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
