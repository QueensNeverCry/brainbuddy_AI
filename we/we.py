import cv2
import os

def extract_frames(video_path, output_folder, fps=5):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    interval = int(frame_rate / fps)

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1
    cap.release()

if __name__ == '__main__':
    extract_frames()