import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ConcentrationClassifier:
    """XGBoost 기반 집중도 분류기 (불균형 데이터 보완)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = None
        
        # 클래스 정보
        self.class_names = {0: '비집중', 1: '주의산만', 2: '집중'}
        self.class_colors = {0: '빨강', 1: '노랑', 2: '초록'}
        
        # 불균형 데이터 처리 방법들
        self.resampling_methods = {
            'smote': SMOTE(random_state=42, k_neighbors=3),
            'borderline': BorderlineSMOTE(random_state=42, k_neighbors=3),
            'adasyn': ADASYN(random_state=42, n_neighbors=3),
            'smote_tomek': SMOTETomek(random_state=42)
        }
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """특징 준비"""
        meta_columns = ['metaid', 'condition', 'posture', 'inst', 'label_3class']
        feature_columns = [col for col in df.columns if col not in meta_columns]
        
        # NaN 값 처리
        df_clean = df[feature_columns].fillna(0)
        
        X = df_clean.values
        y = df['label_3class'].values
        
        print(f"선택된 특징 수: {len(feature_columns)}")
        return X, y, feature_columns
    
    def analyze_class_distribution(self, y: np.ndarray):
        """클래스 분포 분석"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print("\n=== 클래스 분포 분석 ===")
        for cls, count in zip(unique, counts):
            percentage = count / total * 100
            print(f"{self.class_names[cls]}: {count}개 ({percentage:.1f}%)")
        
        # 불균형 비율 계산
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        print(f"불균형 비율: {imbalance_ratio:.2f}:1 {'(심각한 불균형)' if imbalance_ratio > 3 else '(보통 불균형)'}")
        
        return imbalance_ratio
    
    def prepare_data_advanced(self, X: np.ndarray, y: np.ndarray, 
                            method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """간단한 데이터 전처리 (메모리 절약 + 오류 방지)"""
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X.astype(np.float32))  # float32로 메모리 절약
        
        # 불균형 비율 분석
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        print("\n=== 클래스 분포 분석 ===")
        for cls, count in zip(unique, counts):
            percentage = count / total * 100
            print(f"{self.class_names[cls]}: {count}개 ({percentage:.1f}%)")
        
        # 불균형 비율 계산
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        print(f"불균형 비율: {imbalance_ratio:.2f}:1")
        
        # 간단한 리샘플링 적용 (메모리 절약)
        print(f"\n💾 메모리 절약형 리샘플링 적용 중...")
        
        try:
            # 데이터 크기 확인
            data_size_mb = X_scaled.nbytes / (1024 * 1024)
            print(f"현재 데이터 크기: {data_size_mb:.1f} MB")
            
            if data_size_mb > 100:  # 100MB 초과시 리샘플링 제한
                print("⚠️ 데이터가 너무 큼, 클래스 가중치만 적용")
                return X_scaled, y
            
            # 간단한 SMOTE (k_neighbors 최소화)
            smote_simple = SMOTE(
                random_state=42, 
                k_neighbors=min(2, min(counts) - 1),  # 최소 클래스 개수에 맞춤
                sampling_strategy='auto'  # 자동 균형
            )
            
            print("🔄 SMOTE 리샘플링 중...")
            X_balanced, y_balanced = smote_simple.fit_resample(X_scaled, y)
            
            # 결과 확인
            balanced_size_mb = X_balanced.nbytes / (1024 * 1024)
            print(f"리샘플링 후 크기: {balanced_size_mb:.1f} MB")
            
            print(f"✅ 리샘플링 완료:")
            unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
            for cls, count in zip(unique_balanced, counts_balanced):
                print(f"  {self.class_names[cls]}: {count}개")
            
            # 메모리 체크
            if balanced_size_mb > 500:  # 500MB 초과시 샘플 줄이기
                print("⚠️ 리샘플링 데이터가 너무 큼, 샘플링 줄임")
                
                # 각 클래스당 최대 1000개로 제한
                max_samples_per_class = 1000
                indices_to_keep = []
                
                for cls in unique_balanced:
                    cls_indices = np.where(y_balanced == cls)[0]
                    if len(cls_indices) > max_samples_per_class:
                        selected_indices = np.random.choice(
                            cls_indices, 
                            size=max_samples_per_class, 
                            replace=False
                        )
                        indices_to_keep.extend(selected_indices)
                    else:
                        indices_to_keep.extend(cls_indices)
                
                X_balanced = X_balanced[indices_to_keep]
                y_balanced = y_balanced[indices_to_keep]
                
                print(f"📉 샘플링 후 최종 크기: {len(X_balanced)}개")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ 리샘플링 실패: {str(e)}")
            print("🔄 원본 데이터로 진행...")
            
            # 리샘플링 실패시 원본 데이터 사용
            return X_scaled, y
        
        except MemoryError:
            print("❌ 메모리 부족으로 리샘플링 실패")
            print("🔄 원본 데이터로 진행...")
            return X_scaled, y

    
    def train_xgboost_simple(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """간단한 XGBoost 학습 (오류 방지)"""
        
        try:
            # 간단한 XGBoost 설정
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,  # 병렬 처리 끔
                verbosity=0  # 로그 최소화
            )
            
            # 모델 학습
            self.model.fit(X, y)
            self.model_type = 'XGBoost'
            
            # 교차검증
            cv_scores = cross_val_score(self.model, X, y, cv=3)  # 5 → 3으로 줄임
            
            return {
                'model_type': 'XGBoost',
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            print(f"❌ XGBoost 오류: {e}")
            print("🔄 RandomForest로 대체 학습...")
            
            # 대체: RandomForest 사용
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42
            )
            self.model.fit(X, y)
            self.model_type = 'RandomForest'
            
            cv_scores = cross_val_score(self.model, X, y, cv=3)
            return {
                'model_type': 'RandomForest (대체)',
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

    
    def get_feature_importance(self, feature_columns: List[str]) -> Dict:
        """XGBoost 특징 중요도 분석"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # 특징별 중요도 정렬
            feature_importance = list(zip(feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🎯 상위 15개 중요 특징 (XGBoost):")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"  {i+1:2d}. {feature:20s}: {importance:.4f}")
            
            return dict(feature_importance)
        
        return {}
    
    def evaluate_advanced(self, X_test: np.ndarray, y_test: np.ndarray, feature_columns: List[str]) -> Dict:
        """고급 모델 평가 (불균형 데이터 고려)"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # 예측
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # 다양한 메트릭 계산
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 분류 리포트
        report = classification_report(
            y_test, y_pred, 
            target_names=[self.class_names[i] for i in range(3)],
            output_dict=True
        )
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        # 특징 중요도
        feature_importance = self.get_feature_importance(feature_columns)
        
        print(f"\n=== 모델 성능 평가 ===")
        print(f"정확도 (Accuracy): {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """예측 수행"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def save_model(self, filepath: str, feature_columns: List[str]):
        """모델 저장"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'class_names': self.class_names,
            'feature_columns': feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"XGBoost 모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.class_names = model_data['class_names']
        self.feature_columns = model_data.get('feature_columns', [])
        print(f"XGBoost 모델 로드 완료: {filepath}")