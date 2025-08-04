import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from typing import Tuple, Dict, List

class ConcentrationClassifier:
    """JSON 특징 기반 3클래스 집중도 분류기"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.model = None
        self.model_type = None
        
        # 클래스 정보
        self.class_names = {0: '비집중', 1: '주의산만', 2: '집중'}
        self.class_colors = {0: '빨강', 1: '노랑', 2: '초록'}
        
        # 특징 그룹
        self.feature_groups = {
            'static_mean': ['head_yaw_mean', 'head_pitch_mean', 'head_roll_mean', 
                           'gaze_x_mean', 'gaze_y_mean', 'l_EAR_mean', 'r_EAR_mean'],
            'static_std': ['head_yaw_std', 'head_pitch_std', 'head_roll_std',
                          'gaze_x_std', 'gaze_y_std'],
            'dynamic': ['gaze_stability', 'head_stability', 'gaze_jitter',
                       'saccade_frequency', 'fixation_duration', 'distance_change'],
            'behavioral': ['gaze_direction_prob', 'blink_frequency']
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """특징 준비 및 선택"""
        
        # 메타 컬럼 제외
        meta_columns = ['metaid', 'condition', 'posture', 'inst', 'label_3class']
        feature_columns = [col for col in df.columns if col not in meta_columns]
        
        # NaN 값 처리
        df_clean = df[feature_columns].fillna(0)
        
        X = df_clean.values
        y = df['label_3class'].values
        
        print(f"선택된 특징 수: {len(feature_columns)}")
        print("특징 그룹별 개수:")
        for group_name, group_features in self.feature_groups.items():
            count = len([f for f in group_features if f in feature_columns])
            print(f"  {group_name}: {count}개")
        
        return X, y, feature_columns
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, apply_smote: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리 (정규화 + SMOTE)"""
        
        # 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # SMOTE 적용
        if apply_smote:
            X_balanced, y_balanced = self.smote.fit_resample(X_scaled, y)
            print(f"SMOTE 적용 후 클래스 분포:")
            unique, counts = np.unique(y_balanced, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  {self.class_names[cls]}: {count}개")
            return X_balanced, y_balanced
        
        return X_scaled, y
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, tune_hyperparams: bool = True) -> Dict:
        """RandomForest 학습"""
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"최적 RandomForest 파라미터: {best_params}")
            
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            self.model.fit(X, y)
            best_params = {}
        
        self.model_type = 'RandomForest'
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        return {
            'model_type': 'RandomForest',
            'best_params': best_params,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def get_feature_importance(self, feature_columns: List[str]) -> Dict:
        """특징 중요도 분석"""
        if self.model_type == 'RandomForest' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # 특징별 중요도 정렬
            feature_importance = list(zip(feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n상위 10개 중요 특징:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"  {i+1:2d}. {feature:20s}: {importance:.4f}")
            
            return dict(feature_importance)
        
        return {}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, feature_columns: List[str]) -> Dict:
        """모델 평가"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # 예측
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # 정확도
        accuracy = accuracy_score(y_test, y_pred)
        
        # 분류 리포트
        report = classification_report(y_test, y_pred, 
                                     target_names=[self.class_names[i] for i in range(3)],
                                     output_dict=True)
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        
        # 특징 중요도
        feature_importance = self.get_feature_importance(feature_columns)
        
        return {
            'accuracy': accuracy,
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
            'feature_columns': feature_columns,
            'feature_groups': self.feature_groups
        }
        joblib.dump(model_data, filepath)
        print(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.class_names = model_data['class_names']
        self.feature_columns = model_data['feature_columns']
        self.feature_groups = model_data.get('feature_groups', {})
        print(f"모델 로드 완료: {filepath}")
