import os
import cv2
from models.face_crop import crop_face
import mediapipe as mp

def crop_faces_in_existing_images(input_root, output_root, face_detector):
    for root, dirs, files in os.walk(input_root):
        # 이미지 파일만 필터링
        image_files = [f for f in files if f.lower().endswith(".jpg")]
        if not image_files:
            continue

        # 출력 경로 (원본 경로 기준으로 상대 경로 계산)
        rel_path = os.path.relpath(root, input_root)
        output_folder = os.path.join(output_root, rel_path)
        os.makedirs(output_folder, exist_ok=True)

        print(f"[처리중] {rel_path} - 이미지 수: {len(image_files)}")

        for img_file in image_files:
            img_path = os.path.join(root, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ⚠️ 이미지 로드 실패: {img_path}")
                continue

            cropped = crop_face(image, face_detector)
            if cropped is None:
                print(f"  ⚠️ 얼굴 인식 실패: {img_path}")
                continue

            cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_folder, img_file)
            success = cv2.imwrite(save_path, cropped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if success:
                print(f"  ✅ 저장 완료: {save_path}")
            else:
                print(f"  ❌ 저장 실패: {save_path}")

if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    input_root = r"C:\Users\user\Downloads\StudentEngagement"  # 기존 이미지가 저장된 경로
    output_root = r"C:\Student_engagement"      # 얼굴 크롭 결과 저장할 경로

    crop_faces_in_existing_images(input_root, output_root, face_detector)

    face_detector.close()
    print("✅ 얼굴 크롭 작업 완료!")
