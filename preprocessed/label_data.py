import pandas as pd
import os
def load_binary_labels(csv_path,save_path):
    """
    Engagement - Boredom 기준으로 binary label을 생성해 dict로 반환
    key: ClipID (ex: "1100062016")
    value: label (0 or 1)
    """
    df = pd.read_csv(csv_path)

    # 결측값 제거 (필요 시)
    df = df.dropna(subset=["Boredom", "Engagement"])
    df['ClipID'] = df['ClipID'].str.replace(r'\.avi|\.mp4', '', regex=True) #확장자(.mp4, .avi) 제거
    # 이진 라벨 생성
    df["binary_label"] = (df["Engagement"]>= 2).astype(int)

    # 라벨 분포 출력
    label_counts = df["binary_label"].value_counts().sort_index()

    print("✅ 라벨 분포:")
    print(f"  0: {label_counts.get(0, 0)}개")
    print(f"  1: {label_counts.get(1, 0)}개")
    
    # 저장용 DataFrame (ClipID와 라벨만)
    label_df = df[["ClipID", "binary_label"]]
    
    if save_path:
        label_df.to_csv(save_path, index=False)
        print(f"✅ 이진 라벨 CSV 저장 완료: {save_path}")

train_path="../DataSet/Labels/TrainLabels.csv"
val_path= "../DataSet/Labels/ValidationLabels.csv"
all_path ="../DataSet/Labels/AllLabels.csv"

save_train_path = "./pre_labels/pre_TrainLabels.csv"
save_val_path = "./pre_labels/pre_ValidationLabels.csv"
save_all_path = "./pre_labels/pre_AllLabels.csv"

load_binary_labels(train_path,save_train_path)
load_binary_labels(val_path,save_val_path)
load_binary_labels(all_path,save_all_path)