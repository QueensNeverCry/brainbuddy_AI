import os
import cv2
import torch
import pickle
from tqdm import tqdm
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
from models.face_crop import crop_face

"""
특징벡터 추출하고 pkl로 변환
"""

# torchvision용 이미지 변환
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_features_from_folder(frame_folder, model, device, T=100):
    # 해당 폴더가 존재하지 않으면 패스
    if not os.path.exists(frame_folder):
        print(f"[SKIP] 경로 없음 : {frame_folder}")
        return None

    # 이미지 파일 정렬
    img_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(img_files) < T:
        print(f"[SKIP] {frame_folder}: 프레임 부족 ({len(img_files)}/{T})")
        return None

    # 100장 경로 추출
    img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]
    frames = []

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] 이미지 로드 실패: {path}")
            return None
        face_crop = crop_face(img) #얼굴이 없는 경우 프레임 전체 반환
        tensor = transform(face_crop)
        frames.append(tensor)

    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)  # [1, 100, 3, 224, 224]
    features = model(frames_tensor).squeeze(0).cpu()  # [100, 1280]
    return features

def save_features_as_pkl(dataset_link, save_path, device=None, T=100):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNEncoder().to(device)
    model.eval()

    all_features = []
    all_labels = []

    for frame_folder, label in tqdm(dataset_link):
        features = extract_features_from_folder(frame_folder, model, device, T)
        if features is None:
            continue
        all_features.append(features)  # Tensor [100, 1280]
        all_labels.append(torch.tensor(label, dtype=torch.float32))

    with open(save_path, "wb") as f:
        pickle.dump({
            "features": all_features,  # 리스트 [N개 x Tensor [100, 1280]]
            "labels": all_labels       # 리스트 [N개 x Tensor]
        }, f)

    print(f"[✅ 저장 완료] {save_path} | 총 샘플: {len(all_features)}")

if __name__ == "__main__":
    with open("./pickle_labels/train/20_01.pkl", "rb") as f: #(link,label) 가져오기
        dataset_link = pickle.load(f)

    save_features_as_pkl(
        dataset_link,
        save_path="cnn_features/features/train_20_01.pkl", #저장 경로
        T=100
    )
