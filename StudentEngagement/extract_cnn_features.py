import os
import torch
import pickle
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor)
    return feature.squeeze().cpu().numpy()

def extract_features_from_directory(input_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features = {}
    success, fail = 0, 0

    for folder in tqdm(os.listdir(input_dir), desc=f"{os.path.basename(output_path).upper().split('_')[0]} 특징 추출"):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        segment_features = []
        for segment in sorted(os.listdir(folder_path)):
            segment_path = os.path.join(folder_path, segment)
            if not os.path.isdir(segment_path):
                continue

            segment_vector = []
            for file in sorted(os.listdir(segment_path)):
                if not file.lower().endswith(('.jpg', '.png')):
                    continue
                try:
                    img_path = os.path.join(segment_path, file)
                    vector = extract_feature(img_path)
                    segment_vector.append(vector)
                except Exception:
                    continue

            if segment_vector:
                segment_features.append(segment_vector)

        if segment_features:
            features[folder] = segment_features
            success += 1
        else:
            fail += 1

    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"✅ {os.path.basename(output_path).upper().split('_')[0]} 저장 완료: {output_path} (성공: {success} / 실패: {fail} / 전체: {len(os.listdir(input_dir))})")

def main():
    base_dir = "C:/Student_engagement"
    output_dir = "C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features"

    splits = ["train", "test"]
    for split in splits:
        input_path = os.path.join(base_dir, split)
        output_path = os.path.join(output_dir, split, f"{split}_features.pkl")
        extract_features_from_directory(input_path, output_path)

    print("🎉 전체 작업 완료!")

if __name__ == "__main__":
    main()
