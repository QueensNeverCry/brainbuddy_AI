import os
import cv2
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
import multiprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features_from_folder(args):
    frame_folder, label, device_str, T = args

    device = torch.device(device_str)
    model = CNNEncoder().to(device)
    model.eval()

    skip_path_count = 0
    skip_frame_count = 0
    load_fail_count = 0
    success_count = 0

    if not os.path.exists(frame_folder):
        skip_path_count += 1
        return None

    img_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(img_files) < T:
        skip_frame_count += 1
        return None

    img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]
    frames = []

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            load_fail_count += 1
            return None
        # 얼굴 크롭 제거 → 이미지 그대로 transform 적용
        tensor = transform(img)
        frames.append(tensor)

    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
    features = model(frames_tensor).squeeze(0).cpu()

    success_count += 1
    return features, torch.tensor(label, dtype=torch.float32), skip_path_count, skip_frame_count, load_fail_count, success_count


def save_features_as_pkl(dataset_link, save_path, device_str="cuda", T=100, num_workers=4):
    args_list = [(frame_folder, label, device_str, T) for frame_folder, label in dataset_link]

    all_features = []
    all_labels = []

    skip_path_total = 0
    skip_frame_total = 0
    load_fail_total = 0
    success_total = 0

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_features_from_folder, args_list), total=len(args_list), desc="📦 Feature 추출 중"):
            if result is None:
                continue
            features, label, sp, sf, lf, sc = result
            all_features.append(features)
            all_labels.append(label)

            skip_path_total += sp
            skip_frame_total += sf
            load_fail_total += lf
            success_total += sc

    with open(save_path, "wb") as f:
        pickle.dump({
            "features": all_features,
            "labels": all_labels
        }, f)

    print(f"[✅ 저장 완료] {save_path} | 총 샘플: {len(all_features)}")
    print("\n📊 처리 통계:")
    print(f"  [경로 없음] {skip_path_total}")
    print(f"  [프레임 부족] {skip_frame_total}")
    print(f"  [이미지 로드 실패] {load_fail_total}")
    print(f"  [정상 추출 완료] {success_total}")


if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()

    with open("preprocess2/pickle_labels/train/20_01.pkl", "rb") as f:
        dataset_link = pickle.load(f)

    max_workers = min(multiprocessing.cpu_count(), 4)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_features_as_pkl(
        dataset_link,
        save_path="cnn_features/features/train_20_01.pkl",
        device_str=device_str,
        T=100,
        num_workers=max_workers
    )
