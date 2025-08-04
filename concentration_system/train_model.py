import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_classifier import ConcentrationClassifier

def train_json_concentration_model():
    """JSON 특징 기반 3클래스 집중도 모델 학습"""
    
    print("=== JSON 특징 기반 3클래스 집중도 모델 학습 ===")
    
    # 1. 데이터 로드
    print("1단계: 처리된 데이터셋 로드...")
    csv_path = "./processed_data/json_features_3class_dataset.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"데이터 로드 완료: {len(df)}개 샘플")
    except FileNotFoundError:
        print(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 data_processor.py를 실행하여 데이터를 처리하세요.")
        return
    
    # 2. 분류기 초기화 및 특징 준비
    classifier = ConcentrationClassifier()
    X, y, feature_columns = classifier.prepare_features(df)
    
    print(f"특징 벡터 차원: {X.shape[1]}")
    print(f"샘플 수: {len(X)}")
    
    # 클래스 분포 확인
    print(f"\n클래스 분포:")
    for label in [0, 1, 2]:
        count = np.sum(y == label)
        print(f"  {classifier.class_names[label]}: {count}개 ({count/len(y)*100:.1f}%)")
    
    # 3. 학습/테스트 분할 (개인 기반)
    print("\n2단계: 데이터 분할 (개인 기반)...")
    
    # 개인별로 분할하여 일반화 성능 측정
    unique_persons = df['metaid'].unique()
    train_persons = np.random.choice(unique_persons, 
                                   size=int(len(unique_persons) * 0.8), 
                                   replace=False)
    
    train_mask = df['metaid'].isin(train_persons)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"학습 데이터: {len(X_train)}개 ({len(train_persons)}명)")
    print(f"테스트 데이터: {len(X_test)}개 ({len(unique_persons) - len(train_persons)}명)")
    
    # 4. 데이터 전처리
    print("\n3단계: 데이터 전처리...")
    X_train_processed, y_train_processed = classifier.prepare_data(X_train, y_train, apply_smote=True)
    
    # 5. 모델 학습
    print("\n4단계: 모델 학습...")
    
    tune_params = input("하이퍼파라미터 튜닝을 수행하시겠습니까? (y/n): ").lower() == 'y'
    
    # RandomForest 학습
    print("\nRandomForest 학습 중...")
    rf_results = classifier.train_random_forest(X_train_processed, y_train_processed, tune_params)
    print(f"RandomForest CV 점수: {rf_results['cv_mean']:.4f} (+/- {rf_results['cv_std']*2:.4f})")
    
    # 6. 성능 평가
    print("\n5단계: 모델 평가...")
    evaluation = classifier.evaluate(X_test, y_test, feature_columns)
    
    print(f"\n=== 최종 평가 결과 ===")
    print(f"테스트 정확도: {evaluation['accuracy']:.4f}")
    
    # 클래스별 성능
    print(f"\n클래스별 성능:")
    report = evaluation['classification_report']
    for class_name in ['비집중', '주의산만', '집중']:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        print(f"  {class_name:>6}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # 혼동 행렬
    print(f"\n혼동 행렬:")
    cm = evaluation['confusion_matrix']
    print("        예측")
    print("실제    비집중  주의산만  집중")
    class_names_list = ['비집중', '주의산만', '집중']
    for i, true_class in enumerate(class_names_list):
        row = f"{true_class:>6}  "
        for j in range(3):
            row += f"{cm[i][j]:>6}  "
        print(row)
    
    # 7. 모델 저장
    print(f"\n6단계: 모델 저장...")
    model_path = "./json_features_3class_concentration_classifier.pkl"
    classifier.save_model(model_path, feature_columns)
    
    print(f"\n모델 저장 완료: {model_path}")
    print("inference.py로 실시간 테스트를 진행할 수 있습니다.")

if __name__ == "__main__":
    train_json_concentration_model()
