import os
import cv2
import torch
import pickle
from tqdm import tqdm
import mediapipe as mp
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
from multiprocessing import Pool, cpu_count

# torchvision 변환
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 리사이즈 최적화된 얼굴 크롭
def crop_face(img_bgr, face_detector, fallback_to_full=True):
    h, w, _ = img_bgr.shape
    scale = 0.25
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rh, rw, _ = resized_rgb.shape

    results = face_detector.process(resized_rgb)

    if results.detections:
        max_area = 0
        best_bbox = None
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            area = bbox.width * bbox.height
            if area > max_area:
                max_area = area
                best_bbox = bbox

        if best_bbox:
            x1 = max(int((best_bbox.xmin * rw) / scale), 0)
            y1 = max(int((best_bbox.ymin * rh) / scale), 0)
            x2 = min(x1 + int((best_bbox.width * rw) / scale), w)
            y2 = min(y1 + int((best_bbox.height * rh) / scale), h)
            face_crop = img_bgr[y1:y2, x1:x2]
            return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None

# 병렬 작업자 함수
def process_sample(args):
    frame_folder, label, T = args

    if not os.path.exists(frame_folder):
        return None

    img_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if len(img_files) < T:
        return None

    img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]
    frames = []

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                return None
            face = crop_face(img, face_detector)
            tensor = transform(face)
            frames.append(tensor)

    if len(frames) != T:
        return None

    frames_tensor = torch.stack(frames)  # [100, 3, 224, 224]
    return (frames_tensor, label)

def save_features_as_pkl(dataset_link, save_path, device=None, T=100, batch_size=8):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = [(folder, label, T) for folder, label in dataset_link]

    print(f"🛠️ 병렬 전처리 시작: {cpu_count()//16} CPU 사용")
    with Pool(processes=cpu_count()//16) as pool:
        results = list(tqdm(pool.imap(process_sample, tasks), total=len(tasks)))

    # 유효한 샘플 필터링
    valid_results = [(frames, label) for (frames, label) in results if frames is not None]
    print(f"✅ 유효 샘플 수: {len(valid_results)}")

    # CNN 모델 준비
    model = CNNEncoder().to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_results), batch_size), desc="🧠 CNN 배치 추론"):
            batch = valid_results[i:i+batch_size]
            batch_frames = torch.stack([frames for frames, _ in batch])  # [B, 100, 3, 224, 224]
            batch_labels = [label for _, label in batch]

            batch_frames = batch_frames.to(device)
            batch_outputs = model(batch_frames).cpu()  # [B, 100, 1280]

            all_features.extend([feat for feat in batch_outputs])  # 각각 [100, 1280]
            all_labels.extend([torch.tensor(label, dtype=torch.float32) for label in batch_labels])
# 최종 저장 함수
# def save_features_as_pkl(dataset_link, save_path, device=None, T=100):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 병렬 처리 준비
#     tasks = [(folder, label, T) for folder, label in dataset_link]

#     print(f"🛠️ 병렬 처리 시작: {cpu_count()//16} CPU 사용")
#     with Pool(processes=cpu_count()//16) as pool:
#         results = list(tqdm(pool.imap(process_sample, tasks), total=len(tasks)))

#     # 유효한 결과만 필터링
#     valid_results = [(frames, label) for (frames, label) in results if frames is not None]
#     print(f"✅ 유효 샘플 수: {len(valid_results)}")

#     # CNN 모델 로드 및 추론
#     model = CNNEncoder().to(device)
#     model.eval()

#     all_features = []
#     all_labels = []

#     with torch.no_grad():
#         for frames_tensor, label in tqdm(valid_results, desc="🧠 CNN 추론"):
#             input_tensor = frames_tensor.unsqueeze(0).to(device)  # [1, 100, 3, 224, 224]
#             feature = model(input_tensor).squeeze(0).cpu()  # [100, 1280]
#             all_features.append(feature)
#             all_labels.append(torch.tensor(label, dtype=torch.float32))

#     # 저장
#     with open(save_path, "wb") as f:
#         pickle.dump({
#             "features": all_features,
#             "labels": all_labels
#         }, f)

#     print(f"[✅ 저장 완료] {save_path}")
#     print(f"총 유효 샘플: {len(all_features)}")

# 진입점
if __name__ == "__main__":
    with open("preprocess2/pickle_labels/train/20_01.pkl", "rb") as f:
        dataset_link = pickle.load(f)

    save_features_as_pkl(
        dataset_link,
        save_path="cnn_features/features/train_20_01.pkl",
        T=100,
        batch_size=8
    )
