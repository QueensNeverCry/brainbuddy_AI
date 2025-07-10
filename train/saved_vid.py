from frame_extractor import extract_frames
from detection.face_detector import detect_faces
from models.feature_extractor import extract_cnn_features
from models.concent_model import SimpleLSTM
from utils.visualization import show_image
import torch
import torch.nn as nn
import cv2

def build_sequences(faces, T=10):
    return [faces[i:i+T] for i in range(0, len(faces) - T + 1, T)]

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    return frames

video_path = './data/face_video.mp4'
frames = extract_frames(video_path, frame_rate=5)
print(f"추출된 프레임 수: {len(frames)}")
show_image(frames[0], title="첫 프레임")

faces = detect_faces(frames)
print(f"검출된 얼굴 수: {len(faces)}")
show_image(faces[0], title="첫 얼굴")

sequences = build_sequences(faces, T=10)
print(f"시퀀스 수: {len(sequences)}")

feature_sequence = extract_cnn_features(sequences[0])
print(f"시퀀스 특징 벡터 shape: {feature_sequence.shape}")

model = SimpleLSTM()
output = model(feature_sequence)
print(f"예측된 집중도: {output.item():.4f} (0=비집중 ~ 1=집중)")

loss_fn = nn.BCELoss()
label = torch.tensor([0.0]).unsqueeze(1)
loss = loss_fn(output, label)
print(f"Loss: {loss.item():.4f}")