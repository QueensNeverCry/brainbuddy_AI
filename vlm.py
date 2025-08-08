import os
import clip
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd

# 장치 설정
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
except RuntimeError as e:
    print(f"⚠️ GPU 오류, CPU로 재시도: {e}")
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

# 텍스트 프롬프트 정의 (눈동자 + 자세 기반 집중/비집중 표현)
prompts = [
    "a face where the eyes and body are aligned and focused on the same point, indicating concentration",
    "a face where the eyes and body are not aligned or not looking at the same point, indicating lack of focus"
]
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 하위 폴더까지 이미지 포함 여부 확인
def find_all_face_crop_folders(folders):
    valid_folders = []
    for root in folders:
        if not os.path.exists(root):
            continue
        for sub in os.listdir(root):
            sub_path = os.path.join(root, sub)
            if os.path.isdir(sub_path):
                image_files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    valid_folders.append(sub_path)
    return valid_folders

# 라벨링 함수
def label_sequence(folder_path):
    image_features = []
    print(f"\n📂 이미지 확인 중: {folder_path}")
    try:
        file_list = os.listdir(folder_path)
    except Exception as e:
        print(f"❌ 폴더 접근 실패: {folder_path} → {e}")
        return None

    for fname in sorted(file_list):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, fname)
            print(f"  🔍 {fname}")
            try:
                with Image.open(img_path) as img:
                    image = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.encode_image(image)
                        feat /= feat.norm(dim=-1, keepdim=True)
                        image_features.append(feat.cpu().numpy())
            except Exception as e:
                print(f"⚠️ 처리 실패: {img_path} → {e}")
                continue

    if len(image_features) == 0:
        print(f"⛔ 이미지 없음: {folder_path}")
        return None

    avg_feat = torch.tensor(np.mean(np.vstack(image_features), axis=0)).unsqueeze(0).to(device)
    similarity = (avg_feat @ text_features.T).softmax(dim=-1).squeeze().cpu().numpy()
    print(f"✅ 라벨링 완료: {folder_path} → {similarity}")
    return similarity

root_dirs = [
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop"
]


# 전체 실행
results = []
folders = find_all_face_crop_folders(root_dirs)

for folder in folders:
    print(f"\n▶️ 폴더 처리 중: {folder}")
    sim = label_sequence(folder)
    if sim is None:
        continue
    pred_label = int(np.argmax(sim))
    results.append({
        "folder": folder,
        "focused": sim[0],
        "unfocused": sim[1],
        "predicted_label": pred_label
    })

# CSV 저장
if results:
    df = pd.DataFrame(results)
    df.to_csv("vlm_labeled_results_binary.csv", index=False)
    print("\n📁 이진 라벨링 결과 저장 완료 → 'vlm_labeled_results_binary.csv'")
else:
    print("\n⚠️ 저장할 결과가 없습니다. 이미지 경로와 구조를 다시 확인하세요.")
