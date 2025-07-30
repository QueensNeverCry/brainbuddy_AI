import pickle
import torch
import numpy as np

def downsample_features_pkl(pkl_path, output_path, keep_count=30):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    features_list = data["features"]
    labels_list = data["labels"]

    new_features = []
    for feat in features_list:
        if isinstance(feat, torch.Tensor):
            feat_np = feat.numpy()
        else:
            feat_np = feat

        total_frames = feat_np.shape[0]
        if total_frames <= keep_count:
            downsampled = feat_np
        else:
            indices = np.linspace(0, total_frames - 1, num=keep_count, dtype=int)
            downsampled = feat_np[indices]

        # 다시 torch.Tensor로 변환
        new_features.append(torch.tensor(downsampled))

    # 저장
    with open(output_path, "wb") as f:
        pickle.dump({
            "features": new_features,
            "labels": labels_list
        }, f)

    print(f"✅ 저장 완료: {output_path}")
    print(f"🔹 총 샘플 수: {len(new_features)}")

if __name__ == "__main__":
    # original_pkl = ["./features/train_20_01.pkl","./features/train_20_03.pkl","./features/valid_20_01.pkl","./features/valid_20_03.pkl"]
    # output_pkl = ["./features_30/train_20_01.pkl","./features_30/train_20_03.pkl","./features_30/valid_20_01.pkl","./features_30/valid_20_03.pkl"]
    original_pkl =["./features/D_train.pkl","./features/D_val.pkl"]
    output_pkl =["./features_30/D_train.pkl","./features_30/D_val.pkl"]
    for src, dst in zip(original_pkl, output_pkl):
        print(f"🔄 변환 중: {src} → {dst}")
        downsample_features_pkl(src, dst, keep_count=30)