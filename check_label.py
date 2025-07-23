import pickle

# 파일 불러오기
with open('./cnn_features/features/train_20_01.pkl', 'rb') as f:
    data = pickle.load(f)

# 라벨 확인 - 예를 들어 'label' 키가 있을 경우
labels = data['labels']  # 혹은 data[1], data['y'] 등 구조에 따라 다름

# 라벨이 모두 1인지 확인
all_ones = all(label == 1 for label in labels)

print("모든 라벨이 1인가요?", all_ones)
