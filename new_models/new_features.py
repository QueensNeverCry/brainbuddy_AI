import os
import cv2
import torch
import pickle
import numpy as np
import platform
import re
from tqdm import tqdm
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
import multiprocessing

# ─── 경로 변환 함수 ───────────────────────────────────────────────
def windows_path_to_wsl(path: str) -> str:
    """
    WSL 환경일 때 'C:\\...' 경로를 '/mnt/c/...' 형태로 변환.
    Windows 네이티브 Python이면 그대로 반환.
    """
    if platform.system() == "Linux" and re.match(r"^[A-Za-z]:\\", path):
        drive, rest = path.split(":", 1)
        rest = rest.replace("\\", "/")            # 백슬래시를 슬래시로 치환
        return "/mnt/" + drive.lower() + rest     # f-string 대신 문자열 덧셈 사용
    return path

# ─── 안정적인 이미지 읽기 ───────────────────────────────────────────
def robust_imread(path: str):
    """
    1) cv2.imread 시도
    2) Windows UNC prefix(\\\\?\\) 재시도
    3) open + cv2.imdecode 우회
    """
    img = cv2.imread(path)
    if img is not None:
        return img

    if os.name == "nt":
        unc = r"\\\\?\\" + os.path.abspath(path)
        img = cv2.imread(unc)
        if img is not None:
            return img

    try:
        data = open(path, "rb").read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ─── 이미지 전처리(transform) 정의 ────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─── 폴더에서 T개 프레임을 불러와 피처 추출 ─────────────────────────────────
@torch.no_grad()
def extract_features_from_folder(args):
    folder, label, device_str, T = args

    # 폴더 체크
    if not os.path.exists(folder):
        return None

    # 이미지 파일 리스트
    imgs = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if len(imgs) < T:
        return None

    # 모델 로드
    device = torch.device(device_str)
    model = CNNEncoder().to(device)
    model.eval()

    # T개 프레임 읽어 전처리
    frames = []
    for fn in imgs[:T]:
        path = os.path.join(folder, fn)
        path_conv = windows_path_to_wsl(path)
        img_bgr = robust_imread(path_conv)
        if img_bgr is None:
            return None
        frames.append(transform(img_bgr))

    # (1, T, C, H, W) 형태로 모델에 입력
    inp = torch.stack(frames).unsqueeze(0).to(device)
    feat = model(inp).squeeze(0).cpu()  # (feature_dim,) 또는 (T, feature_dim)
    return feat, torch.tensor(label, dtype=torch.float32)

# ─── 멀티프로세싱으로 피처 추출 후 .pkl 저장 ───────────────────────────────
def save_features_as_pkl(dataset_link, save_path,
                         device_str="cuda", T=30, num_workers=4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    args_list = [(p, l, device_str, T) for p, l in dataset_link]
    features, labels = [], []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(extract_features_from_folder, args_list),
                        total=len(args_list),
                        desc=f"📦 Extract → {os.path.basename(save_path)}"):
            if res is None:
                continue
            fvec, lbl = res
            features.append(fvec)
            labels.append(lbl)

    with open(save_path, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)
    print(f"[✅ 저장 완료] {save_path} | 샘플 수: {len(features)}")

# ─── 메인 실행부 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1) 처리할 세 개의 입력 폴더
    input_dirs = [
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop", #파일 다운로드 경로를 영어로 바꾸어야함.
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop"
    ]

    # 2) 각 폴더에서 (폴더경로, 라벨) 리스트 생성
    all_dataset_link = []
    for ts_dir in input_dirs:
        class_names = sorted(os.listdir(ts_dir))
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for name in class_names:
            folder = os.path.join(ts_dir, name)
            if os.path.isdir(folder):
                all_dataset_link.append((folder, label_to_idx[name]))
    print(f"전체 샘플 폴더 수: {len(all_dataset_link)}")

    # 3) 피처 저장 설정
    save_path = r"C:/Users/user/Desktop/brainbuddy_AI/cnn_features/features/0801_TS_features.pkl"
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = min(multiprocessing.cpu_count(), 4)
    T = 30  # 프레임 수

    # 4) 피처 추출 및 단일 .pkl 저장
    save_features_as_pkl(all_dataset_link, save_path,
                         device_str=device_str, T=T, num_workers=num_workers)
