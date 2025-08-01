#JSON의 값을 특징 벡터로 변환하는 작업
#이를 라벨링
import os
import pandas as pd
from extract_features import process_json_folder, load_json, extract_features, compute_dynamic_features
from tqdm import tqdm

# 📁 데이터 루트
data_root = r"C:/eye_dataset/train"
output_csv_path = r"C:/eye_dataset/all_features.csv"

# 📌 집중도 매핑 딕셔너리
label_map = {
    "F": "focused",
    "S": "sleepy",
    "D": "inattentive",
    "A": "declining",
    "N": "negligent"
}

label_index = {
    "F": 0,
    "S": 1,
    "D": 2,
    "A": 3,
    "N": 4
}

# 전체 데이터 저장용 리스트
all_data = []

all_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

# tqdm으로 폴더 진행 표시
for folder_name in tqdm(all_folders, desc="📂 폴더별 처리"):
    folder_path = os.path.join(data_root, folder_name)
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    folder_features = []

    for idx, file in enumerate(json_files):
        try:
            data = load_json(os.path.join(folder_path, file))
            anno = data["Annotations"]
            features = extract_features(anno)

            # 공통 메타정보 추가
            features["folder_name"] = folder_name
            features["frame_idx"] = idx

            # 집중도 라벨 추출 및 매핑
            try:
                attention_code = folder_name.split("_")[7]  # 8번째 요소
                features["attention_label"] = label_map.get(attention_code, "unknown")
                features["attention_idx"] = label_index.get(attention_code, -1)
            except Exception as e:
                features["attention_label"] = "unknown"
                features["attention_idx"] = -1
                print(f"⚠️ 라벨 추출 오류 - {folder_name}: {e}")

            folder_features.append(features)

        except Exception as e:
            print(f"⚠️ JSON 처리 오류 - {folder_name}/{file}: {e}")
            continue

    if folder_features:
        df = pd.DataFrame(folder_features)
        df = compute_dynamic_features(df)
        all_data.append(df)

# 📊 통합 DataFrame 저장
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ 모든 feature를 통합 저장 완료: {output_csv_path}")
    print(f"총 데이터 수 (프레임 수): {len(full_df)}")
    print(f"📁 고유 클래스(label): {full_df['attention_label'].unique().tolist()}")
else:
    print("❌ 처리할 유효한 JSON이 없습니다.")
