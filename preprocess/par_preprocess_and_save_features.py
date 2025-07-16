import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.face_detector import extract_face
from models.feature_extractor import extract_cnn_features
import mediapipe as mp
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
import gc

class OptimizedDatasetProcessor:
    def __init__(self, save_dir, T=300, device=None, batch_size=16, num_workers=4):
        self.save_dir = save_dir
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 통계 정보
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        
        # 스레드 로컬 저장소를 위한 변수
        self.thread_local = threading.local()
        
    def get_face_mesh(self):
        """스레드별로 독립적인 FaceMesh 인스턴스 반환"""
        if not hasattr(self.thread_local, 'face_mesh'):
            self.thread_local.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
        return self.thread_local.face_mesh
    
    def safe_extract_face(self, img, img_path):
        """안전한 얼굴 추출 함수"""
        try:
            # 이미지 유효성 검사
            if img is None or img.size == 0:
                return None
                
            # 스레드별 FaceMesh 인스턴스 사용
            face_mesh = self.get_face_mesh()
            
            # MediaPipe 에러 방지를 위한 이미지 복사
            img_copy = img.copy()
            
            # 얼굴 추출 시도
            face = extract_face(img_copy, face_mesh)
            
            return face
            
        except Exception as e:
            # MediaPipe 관련 에러를 조용히 처리
            if "timestamp" in str(e).lower() or "graph" in str(e).lower():
                return None
            else:
                print(f"[ERROR] Face extraction {img_path}: {str(e)[:100]}...")
                return None
    
    def load_and_process_images(self, img_paths):
        """이미지 로드 및 얼굴 추출을 순차적으로 처리"""
        faces = []
        last_valid_face = None
        successful_extractions = 0
        
        for i, img_path in enumerate(img_paths):
            try:
                # 이미지 로드
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 얼굴 추출
                face = self.safe_extract_face(img, img_path)
                
                if face is not None:
                    faces.append(face)
                    last_valid_face = face
                    successful_extractions += 1
                elif last_valid_face is not None:
                    # 이전 유효한 얼굴 재사용
                    faces.append(last_valid_face)
                else:
                    # 얼굴을 찾을 수 없고 이전 얼굴도 없는 경우
                    continue
                
                # 충분한 얼굴을 얻었으면 중단
                if len(faces) >= self.T:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Processing {img_path}: {str(e)[:100]}...")
                continue
        
        return faces, successful_extractions
    
    def process_sample(self, sample_data):
        """단일 샘플 처리"""
        sample_idx, (frame_folder, label) = sample_data
        feature_save_path = os.path.join(self.save_dir, f"sample_{sample_idx}.pt")
        
        # 이미 처리된 경우 건너뜀
        if os.path.exists(feature_save_path):
            return "exists"
        
        try:
            # 이미지 파일 목록 가져오기
            img_files = sorted([
                f for f in os.listdir(frame_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except FileNotFoundError:
            return "skip"
        
        if len(img_files) < self.T:
            return "skip"
        
        # 필요한 프레임만 선택
        img_paths = [os.path.join(frame_folder, f) for f in img_files[:self.T]]
        
        # 이미지 처리 및 얼굴 추출
        faces, successful_extractions = self.load_and_process_images(img_paths)
        
        if len(faces) < self.T:
            print(f"[SKIP] {frame_folder}: failed to extract enough faces ({len(faces)} < {self.T}, successful: {successful_extractions})")
            return "skip"
        
        # T개로 맞춤
        faces = faces[:self.T]
        
        try:
            # GPU에서 CNN 특징 추출
            with torch.no_grad():
                features = extract_cnn_features(faces, self.device)
                
                # CPU로 이동하여 저장
                torch.save({
                    'features': features.cpu(),
                    'label': torch.tensor([label], dtype=torch.float32)
                }, feature_save_path)
            
            return "success"
            
        except Exception as e:
            print(f"[ERROR] Feature extraction {frame_folder}: {str(e)[:100]}...")
            return "error"
    
    def process_batch_samples(self, batch_data):
        """배치 단위로 샘플들을 처리 (멀티스레딩 최소화)"""
        results = []
        
        # 작은 배치 크기로 멀티스레딩 사용
        max_workers = min(self.num_workers, 2)  # MediaPipe 충돌 방지를 위해 최대 2개 스레드
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for sample_data in batch_data:
                future = executor.submit(self.process_sample, sample_data)
                futures.append(future)
            
            # 타임아웃 설정으로 무한 대기 방지
            for future in as_completed(futures, timeout=300):  # 5분 타임아웃
                try:
                    result = future.result(timeout=60)  # 1분 타임아웃
                    results.append(result)
                    
                    if result == "success":
                        self.processed_count += 1
                    elif result == "skip":
                        self.skipped_count += 1
                    elif result == "error":
                        self.error_count += 1
                        
                except Exception as e:
                    print(f"[ERROR] Future execution failed: {str(e)[:100]}...")
                    self.error_count += 1
                    results.append("error")
        
        return results
    
    def process_dataset(self, dataset_link):
        """전체 데이터셋 처리"""
        print(f"Processing {len(dataset_link)} samples...")
        
        # 더 작은 배치 크기로 안정성 향상
        batch_size = min(self.batch_size, 8)  # MediaPipe 안정성을 위해 작은 배치 사용
        
        try:
            for i in tqdm(range(0, len(dataset_link), batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, len(dataset_link))
                batch_data = [(i + j, dataset_link[i + j]) for j in range(batch_end - i)]
                
                # 배치 처리
                self.process_batch_samples(batch_data)
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 가비지 컬렉션으로 메모리 정리
                gc.collect()
                
                # 진행 상황 출력
                if i % (batch_size * 10) == 0:
                    print(f"Progress: {i}/{len(dataset_link)} samples processed")
                    print(f"Stats - Success: {self.processed_count}, Skip: {self.skipped_count}, Error: {self.error_count}")
                    
        except KeyboardInterrupt:
            print("\n[INFO] Processing interrupted by user")
            print(f"Processed so far - Success: {self.processed_count}, Skip: {self.skipped_count}, Error: {self.error_count}")
            return
        
        print(f"Processing completed!")
        print(f"Successfully processed: {self.processed_count}")
        print(f"Skipped: {self.skipped_count}")
        print(f"Errors: {self.error_count}")

def preprocess_dataset_optimized(dataset_link, save_dir, T=300, device=None, batch_size=8, num_workers=2):
    """최적화된 데이터셋 전처리 함수"""
    
    # GPU 메모리 최적화 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    processor = OptimizedDatasetProcessor(
        save_dir=save_dir,
        T=T,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    start_time = time.time()
    
    try:
        processor.process_dataset(dataset_link)
        end_time = time.time()
        
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        if processor.processed_count > 0:
            print(f"Average time per sample: {(end_time - start_time) / processor.processed_count:.2f} seconds")
            
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        print(f"Processed so far - Success: {processor.processed_count}, Skip: {processor.skipped_count}, Error: {processor.error_count}")

# 실행 부분
if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 데이터 로드
    try:
        with open(os.path.join(base_dir, "train_link.pkl"), "rb") as f:
            train_link = pickle.load(f)
        with open(os.path.join(base_dir, "val_link.pkl"), "rb") as f:
            val_link = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Data file not found: {e}")
        exit(1)
    
    # GPU 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 안정성을 위한 보수적인 설정
    batch_size = 8
    num_workers = 2
    
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    
    # 훈련 데이터 전처리
    print("=" * 50)
    print("Processing training data...")
    try:
        preprocess_dataset_optimized(
            train_link, 
            save_dir="preprocessed_features/train_data", 
            T=300,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers
        )
    except Exception as e:
        print(f"[ERROR] Training data processing failed: {e}")
    
    # 검증 데이터 전처리
    print("=" * 50)
    print("Processing validation data...")
    try:
        preprocess_dataset_optimized(
            val_link, 
            save_dir="preprocessed_features/val_data", 
            T=300,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers
        )
    except Exception as e:
        print(f"[ERROR] Validation data processing failed: {e}")
    
    print("=" * 50)
    print(f"총 학습 데이터 수: {len(train_link)}개")
    print(f"총 검증 데이터 수: {len(val_link)}개")
    print("전처리 완료!")