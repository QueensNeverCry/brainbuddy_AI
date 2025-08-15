import os
import cv2
import json
import time
import glob
import argparse
from collections import deque
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

# 학습 스크립트와 동일 모델 (파일명이 python2_1.py면 import 라인만 바꿔줘)
from clip2_1 import CNN_LSTM

# ========= 전처리(eval과 동일) =========
def build_eval_tfms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def load_threshold(default=0.5, path="threshold.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return default

# ========= 얼굴 검출 (MediaPipe → 실패시 Haar) =========
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    mp_face = None
    USE_MP = False

HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_bbox(frame_bgr):
    """가장 큰 얼굴 bbox 반환: (x, y, w, h) / 없으면 None"""
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
    if len(faces) == 0: return None
    x, y, ww, hh = max(faces, key=lambda b: b[2]*b[3])
    return int(x), int(y), int(ww), int(hh)

def tight_square_crop(frame_bgr, bbox, box_scale=1.6):
    """bbox 기준 정사각 타이트 크롭. 경계 밖은 복제 패딩."""
    x, y, fw, fh = bbox
    h, w = frame_bgr.shape[:2]
    s = int(max(fw, fh) * box_scale)
    s = max(s, int(max(fw, fh)*1.2))  # 너무 작게 안 잘리도록
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

# ========= 텐서 변환 =========
def to_tensor_from_bgr(frame_bgr, tfms, auto_zoom=False, first_scale=1.6):
    """스트림 버퍼용: auto_zoom이면 첫 배율로 타이트 크롭 후 변환(빠름)"""
    if auto_zoom:
        bbox = detect_face_bbox(frame_bgr)
        if bbox is not None:
            frame_bgr = tight_square_crop(frame_bgr, bbox, box_scale=first_scale)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return tfms(Image.fromarray(rgb))

def tensors_for_scales(frame_bgr, tfms, scales):
    """멀티스케일용: 각 배율로 타이트 크롭 → 텐서 리스트"""
    bbox = detect_face_bbox(frame_bgr)
    if bbox is None:
        return [tfms(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))]
    tens = []
    for s in scales:
        crop = tight_square_crop(frame_bgr, bbox, box_scale=s)
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tens.append(tfms(Image.fromarray(rgb)))
    return tens

# ========= 소스별 프레임 이터레이터 =========
def iter_webcam(seq_len, tfms, cam_index=0, auto_zoom=False, first_scale=1.6):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened(): raise RuntimeError(f"웹캠 열기 실패 (index={cam_index})")
    buf = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            ten = to_tensor_from_bgr(frame, tfms, auto_zoom=auto_zoom, first_scale=first_scale)
            buf.append(ten)
            if len(buf) > seq_len: buf.pop(0)
            if len(buf) == seq_len:
                seq = torch.stack(buf, dim=0).unsqueeze(0)  # (1,T,3,H,W)
                yield frame, seq
    finally:
        cap.release()

def iter_video(path, seq_len, tfms, auto_zoom=False, first_scale=1.6):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"비디오 열기 실패: {path}")
    buf = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            ten = to_tensor_from_bgr(frame, tfms, auto_zoom=auto_zoom, first_scale=first_scale)
            buf.append(ten)
            if len(buf) > seq_len: buf.pop(0)
            if len(buf) == seq_len:
                seq = torch.stack(buf, dim=0).unsqueeze(0)
                yield frame, seq
    finally:
        cap.release()

def iter_folder(path, seq_len, tfms, auto_zoom=False, first_scale=1.6):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(path, e)))
    files = sorted(files)
    if len(files) < seq_len: raise RuntimeError(f"프레임 수 부족: {len(files)} < {seq_len}")
    k = 0
    while k + seq_len <= len(files):
        chunk = files[k:k+seq_len]; k += seq_len
        frames = []
        for fp in chunk:
            img = cv2.imread(fp)
            if img is None: print(f"[경고] 읽기 실패: {fp}"); continue
            frames.append(img)
        if len(frames) < seq_len: continue
        tens = [to_tensor_from_bgr(f, tfms, auto_zoom=auto_zoom, first_scale=first_scale) for f in frames]
        seq = torch.stack(tens, dim=0).unsqueeze(0)
        yield frames[0], seq

def get_iterator(source, path, seq_len, tfms, cam_index=0, auto_zoom=False, first_scale=1.6):
    if source == "webcam": return iter_webcam(seq_len, tfms, cam_index=cam_index, auto_zoom=auto_zoom, first_scale=first_scale)
    if source == "video":  return iter_video(path, seq_len, tfms, auto_zoom=auto_zoom, first_scale=first_scale)
    if source == "folder": return iter_folder(path, seq_len, tfms, auto_zoom=auto_zoom, first_scale=first_scale)
    raise ValueError("source must be one of [webcam, video, folder]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, choices=["webcam","video","folder"])
    ap.add_argument("--path", default="")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18","efficientnet_b0"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.3)

    # 판정/안정화 (기본값 조정)
    ap.add_argument("--threshold", type=float, default=0.25, help="cut-off (기본 0.25; 미지정 시 이 값 사용)")
    ap.add_argument("--debounce-k", type=int, default=2, help="라벨 변경에 필요한 연속 동일 예측 수(k). 1이면 해제")
    ap.add_argument("--logit-bias", type=float, default=0.2, help="시그모이드 전에 로짓에 더할 상수(민감도↑)")

    # 오토줌 & 멀티스케일 (기본값 조정)
    ap.add_argument("--auto-zoom", action="store_true", default=True, help="얼굴 크기 정규화(멀티스케일 포함)")
    ap.add_argument("--zoom-scales", type=str, default="1.2,1.4,1.6,1.8",
                    help="멀티스케일 배율 리스트(작을수록 타이트)")
    ap.add_argument("--first-scale", type=float, default=1.2, help="스트림 버퍼용 기본 배율(빠름)")
    ap.add_argument("--sweep-frames", type=int, default=5, help="스윕 시 마지막 몇 프레임을 교체해서 재평가할지")

    # 표시/저장
    ap.add_argument("--amb-low", type=float, default=0.45)
    ap.add_argument("--amb-high", type=float, default=0.55)
    ap.add_argument("--dump-dir", default="dump_frames")
    args = ap.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델
    model = CNN_LSTM(
        backbone=args.backbone, hidden=args.hidden, num_layers=args.num_layers,
        bidirectional=(not args.unidirectional), dropout=args.dropout
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 임계치 (기본 0.25)
    TH = args.threshold

    # 전처리 & 이터레이터
    tfms = build_eval_tfms(args.img_size)
    iterator = get_iterator(
        source=args.source, path=args.path, seq_len=args.seq_len, tfms=tfms,
        cam_index=args.cam_index, auto_zoom=args.auto_zoom, first_scale=args.first_scale
    )
    scales = [float(s) for s in args.zoom_scales.split(",") if s.strip()]

    # 디바운스
    last_label, run_len, stable_label = 0, 0, 0

    t0 = time.time(); cnt = 0
    print(f"[INFO] device={device.type} source={args.source} TH={TH:.2f} auto_zoom={args.auto_zoom} "
          f"scales={scales} debounce_k={args.debounce_k} logit_bias={args.logit_bias} sweep_frames={args.sweep_frames}")
    print("[INFO] 키: ESC 종료 | S 스냅샷 | A 애매구간 표시 토글")

    with torch.inference_mode():
        amb_only = False
        for frame_bgr, seq in iterator:
            cnt += 1
            fps = cnt / max(1e-6, (time.time() - t0))

            # 1) 기본 추론 (+ logit bias)
            seq = seq.to(device, non_blocking=True)
            logit = model(seq).float()
            logit = logit + args.logit_bias
            prob = torch.sigmoid(logit).item()

            # 2) 확률이 낮으면 멀티스케일 스윕(트리거 0.60로 상향) — 마지막 M프레임 교체
            if args.auto_zoom and prob < 0.60:
                best = prob
                bbox = detect_face_bbox(frame_bgr)
                if bbox is not None:
                    for s in scales:
                        crop = tight_square_crop(frame_bgr, bbox, box_scale=s)
                        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        patch = tfms(Image.fromarray(rgb)).unsqueeze(0)  # (1,3,H,W)

                        seq_sw = seq.clone()
                        m = max(1, min(args.sweep_frames, seq_sw.size(1)))  # 마지막 M프레임
                        seq_sw[0, -m:] = patch.repeat(m, 1, 1, 1)

                        logit_sw = model(seq_sw).float() + args.logit_bias
                        p = torch.sigmoid(logit_sw).item()
                        if p > best: best = p
                    prob = best

            raw_label = 1 if prob >= TH else 0

            # 3) 디바운스
            if args.debounce_k and args.debounce_k > 1:
                if raw_label == last_label: run_len += 1
                else: last_label, run_len = raw_label, 1
                if run_len >= args.debounce_k: stable_label = raw_label
                label = stable_label
            else:
                label = raw_label

            # 4) 표시
            amb = (args.amb_low <= prob <= args.amb_high)
            txt = f"p={prob:.3f}  TH={TH:.2f}  y={label}  FPS={fps:.1f}"
            color = (0,255,0) if label==1 else (0,0,255)
            cv2.putText(frame_bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if amb: cv2.rectangle(frame_bgr, (8,8), (300,44), (0,255,255), 2)
            if not (amb_only and not amb):
                cv2.imshow("focus-realtime", frame_bgr)

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k in (ord('s'), ord('S')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                outp = os.path.join(args.dump_dir, f"{ts}_p{prob:.3f}_y{label}.jpg")
                cv2.imwrite(outp, frame_bgr); print(f"[Saved] {outp}")
            elif k in (ord('a'), ord('A')):
                amb_only = not amb_only
                print(f"[Toggle] 애매구간만 보기: {amb_only}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

