import cv2
from PIL import Image
from torchvision import transforms

# Optional mediapipe (graceful fallback to Haar cascade)
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    mp_face = None
    USE_MP = False

HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def build_eval_tfms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def detect_face_bbox(frame_bgr):
    """Return (x,y,w,h) for the largest face, or None."""
    h, w = frame_bgr.shape[:2]
    if USE_MP:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if res.detections:
            boxes = []
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                x, y, ww, hh = int(bb.xmin*w), int(bb.ymin*h), int(bb.width*w), int(bb.height*h)
                boxes.append((x, y, ww, hh))
            if boxes:
                x, y, ww, hh = max(boxes, key=lambda b: b[2]*b[3])
                return max(0,x), max(0,y), max(1,ww), max(1,hh)
    # fallback: Haar
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40,40))
    if len(faces) == 0:
        return None
    x, y, ww, hh = max(faces, key=lambda b: b[2]*b[3])
    return int(x), int(y), int(ww), int(hh)

def tight_square_crop(frame_bgr, bbox, box_scale=1.6):
    x, y, fw, fh = bbox
    h, w = frame_bgr.shape[:2]
    s = int(max(fw, fh) * box_scale)
    s = max(s, int(max(fw, fh)*1.2))
    cx, cy = x + fw//2, y + fh//2
    x0, y0 = cx - s//2, cy - s//2
    x1, y1 = cx + s//2, cy + s//2
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        frame_bgr = cv2.copyMakeBorder(frame_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = frame_bgr[max(0,y0):y1, max(0,x0):x1]
    return crop if crop.size else frame_bgr

def to_tensor_from_bgr(frame_bgr, tfms, auto_zoom=False, first_scale=1.6):
    if auto_zoom:
        bbox = detect_face_bbox(frame_bgr)
        if bbox is not None:
            frame_bgr = tight_square_crop(frame_bgr, bbox, box_scale=first_scale)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return tfms(Image.fromarray(rgb))
