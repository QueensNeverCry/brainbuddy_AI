# 마진 넣어서 잘라보기
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

image_path = r"C:/Users/user/Downloads/Student-engagement-dataset/Student-engagement-dataset/Engaged/segment_8/0144.jpg"
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_h, img_w = img_rgb.shape[:2]

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
results = face_detector.process(img_rgb)

def apply_margin_bbox(xmin, ymin, width, height, img_w, img_h, margin_ratio):
    x1 = int((xmin - margin_ratio * width) * img_w)
    y1 = int((ymin - margin_ratio * height) * img_h)
    x2 = int((xmin + (1 + margin_ratio) * width) * img_w)
    y2 = int((ymin + (1 + margin_ratio) * height) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

# Crop and show
if results.detections:
    bbox = results.detections[0].location_data.relative_bounding_box
    plt.figure(figsize=(15, 5))
    for i, margin in enumerate([0.1, 0.15, 0.2]):
        x1, y1, x2, y2 = apply_margin_bbox(bbox.xmin, bbox.ymin, bbox.width, bbox.height, img_w, img_h, margin)
        crop = img_rgb[y1:y2, x1:x2]
        plt.subplot(1, 3, i+1)
        plt.imshow(crop)
        plt.title(f"margin={margin}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()