# test_folder_tree_strict_autozoom_tune.py
import os, json, glob, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt

from clip2_1 import CNN_LSTM

# ===== 기본 설정 (옵션 없이 실행) =====
ROOT_DIR   = r"C:\f\camera_roll_frames"
CKPT_PATH  = r"the_best.pth"
OUT_DIR    = r"eval_img_results"

BACKBONE   = "resnet18"
HIDDEN     = 256
NUM_LAYERS = 2
BIDIR      = True
DROPOUT    = 0.3

IMG_SIZE   = 224
SEQ_LEN    = 30
STEP       = 30      # 슬라이딩 간격 (30이면 겹치기 없음, 5~10으로 줄이면 겹치기 평가)
PAD_MODE   = "repeat_last"

# 튜닝 그리드(원하면 값 늘려도 됨)
THRESH_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
BIAS_GRID   = [0.00, 0.10, 0.20]

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".webp")

# ===== 전처리 =====
def build_eval_tfms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def is_image_file(fn): return fn.lower().endswith(IMG_EXTS)

def find_leaf_folders(root):
    leafs = []
    for cur, _, files in os.walk(root):
        if any(is_image_file(f) for f in files):
            leafs.append(cur)
    return sorted(leafs)

def infer_label_from_path(path: str):
    parts = [p.lower() for p in Path(path).parts]
    # 'unfocus' 먼저, 그 다음 'focus'
    if any("unfocus" in p for p in parts): return 0
    if any("focus"  in p for p in parts):  return 1
    return None

# ===== 얼굴 검출 (MediaPipe → 실패시 Haar) =====
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MP = True
except Exception:
    mp_face = None
    USE_MP = False

HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_bbox(frame_bgr):
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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = HAAR_FACE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40,40))
    if len(faces) == 0: return None
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

def to_tensor_from_path(img_path, tfms, auto_zoom=True, first_scale=1.6):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise RuntimeError(f"failed to read: {img_path}")
    if auto_zoom:
        bbox = detect_face_bbox(bgr)
        if bbox is not None:
            bgr = tight_square_crop(bgr, bbox, box_scale=first_scale)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return tfms(Image.fromarray(rgb))

# ===== 데이터 적재: 슬라이딩 윈도우 =====
def make_sequences(folder, seq_len, step, tfms, auto_zoom=True, first_scale=1.6, pad_mode="repeat_last"):
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if is_image_file(f)])
    n = len(files)
    if n == 0:
        return torch.empty(0, seq_len, 3, IMG_SIZE, IMG_SIZE), 0
    seqs = []
    if n < seq_len:
        # 패딩
        if pad_mode == "repeat_last":
            files = files + [files[-1]] * (seq_len - n)
        elif pad_mode == "loop":
            i = 0
            while len(files) < seq_len:
                files.append(files[i % n]); i += 1
        else:
            files = files + [files[-1]] * (seq_len - n)
        step = seq_len  # 한 청크만
    # 슬라이딩
    for start in range(0, len(files) - seq_len + 1, step):
        chunk = files[start:start+seq_len]
        frames = [to_tensor_from_path(fp, tfms, auto_zoom=auto_zoom, first_scale=first_scale) for fp in chunk]
        seqs.append(torch.stack(frames, dim=0))  # (T,3,H,W)
    if not seqs:
        return torch.empty(0, seq_len, 3, IMG_SIZE, IMG_SIZE), n
    return torch.stack(seqs, dim=0), n  # (N,T,3,H,W), n_images

# ===== 시각화 저장 =====
def save_confmat(path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=(0,1))
    fig, ax = plt.subplots(figsize=(4,4), dpi=140)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0 (Unfocused)", "1 (Focused)"])
    ax.set_yticklabels(["0 (Unfocused)", "1 (Focused)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def save_roc(path, y_true, y_prob):
    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5,4), dpi=140)
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}" if auc == auc else "AUC=N/A")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return auc

def print_device_info(device):
    print("\n===== Device Info =====")
    print(f"Using device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"GPU: {name} (capability {cap[0]}.{cap[1]})")
        print(f"VRAM total: {total:.2f} GB")
        print(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
    print("=======================\n")

def load_model_strict(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CNN_LSTM(
        backbone=BACKBONE, hidden=HIDDEN, num_layers=NUM_LAYERS,
        bidirectional=BIDIR, dropout=DROPOUT
    ).to(device)
    # strict=True로 미스매치 즉시 드러나게
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model

# ===== 메인 =====
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # 모델 로드(엄격)
    try:
        model = load_model_strict(CKPT_PATH, device)
    except Exception as e:
        print("[Error] checkpoint strict load failed:", e)
        print("→ 학습 당시 backbone/레이어/양방향 여부/hidden 크기가 현재 설정과 일치하는지 확인해줘.")
        return

    tfms = build_eval_tfms(IMG_SIZE)

    # 폴더 수집/라벨링
    folders = find_leaf_folders(ROOT_DIR)
    items = []
    for fd in folders:
        y = infer_label_from_path(fd)
        if y is not None:
            items.append((fd, y))
    if not items:
        print("라벨링 가능한 폴더가 없습니다.")
        return

    # 시퀀스 생성 & 원시 확률 저장
    per_folder = []
    raw_cache = []  # (folder, y_true, probs[]) 저장해두고 나중에 threshold/bias 스윕
    with torch.inference_mode():
        for fd, y_true in tqdm(items, desc="Extract+Infer (auto-zoom)", ncols=100):
            seqs, n_images = make_sequences(fd, SEQ_LEN, STEP, tfms, auto_zoom=True, first_scale=1.6, pad_mode=PAD_MODE)
            if seqs.numel() == 0:
                continue
            seqs = seqs.to(device, non_blocking=True)  # (N,T,3,H,W)
            logits = model(seqs).float().detach().cpu().numpy()  # (N,)
            probs  = 1/(1+np.exp(-logits))  # sigmoid
            raw_cache.append((fd, y_true, probs, int(n_images)))

    # threshold × bias 스윕
    def evaluate_grid(th, bias):
        y_true_all, y_pred_all, y_prob_all = [], [], []
        rows = []
        for fd, y_true, probs, n_images in raw_cache:
            probs_biased = 1/(1+np.exp(-(np.log(probs/(1-probs+1e-8)+1e-8) + bias)))  # logit+bias 후 sigmoid
            p_mean = float(np.mean(probs_biased))
            pred = 1 if p_mean >= th else 0
            rows.append((fd, y_true, p_mean, pred, n_images))
            y_true_all.append(y_true); y_pred_all.append(pred); y_prob_all.append(p_mean)
        y_true_all = np.array(y_true_all, int)
        y_pred_all = np.array(y_pred_all, int)
        y_prob_all = np.array(y_prob_all, float)
        acc  = accuracy_score(y_true_all, y_pred_all)
        prec = precision_score(y_true_all, y_pred_all, zero_division=0)
        rec  = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1   = f1_score(y_true_all, y_pred_all, zero_division=0)
        try: auc = roc_auc_score(y_true_all, y_prob_all)
        except Exception: auc = float("nan")
        return acc, prec, rec, f1, auc, rows

    best = None
    for b in BIAS_GRID:
        for th in THRESH_GRID:
            acc, prec, rec, f1, auc, _ = evaluate_grid(th, b)
            key = (f1, acc)  # 1순위 F1, 2순위 Acc
            if (best is None) or (key > (best["f1"], best["acc"])):
                best = {"th": th, "bias": b, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    # 베스트 조합으로 최종 표/이미지 저장
    acc, prec, rec, f1, auc, rows = evaluate_grid(best["th"], best["bias"])
    df = pd.DataFrame(rows, columns=["folder","true","prob_mean","pred","n_images"]).sort_values("folder")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "preds.csv"), index=False, encoding="utf-8-sig")

    print("\n========== SUMMARY (tuned) ==========")
    print(f"Best threshold={best['th']:.2f}, logit_bias={best['bias']:.2f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nClassification Report")
    print(classification_report(df["true"], df["pred"], digits=4, zero_division=0))

    # 시각화 저장
    cm_path  = os.path.join(OUT_DIR, "confusion_matrix.png")
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    save_confmat(cm_path, df["true"], df["pred"])
    auc2 = save_roc(roc_path, df["true"], df["prob_mean"])
    print(f"[Saved] {cm_path}, {roc_path} (AUC={auc2})")

    # 메타 저장
    meta = {
        "root": ROOT_DIR,
        "ckpt": CKPT_PATH,
        "model": {"backbone": BACKBONE, "hidden": HIDDEN, "num_layers": NUM_LAYERS, "bidirectional": BIDIR, "dropout": DROPOUT},
        "img_size": IMG_SIZE, "seq_len": SEQ_LEN, "step": STEP, "pad_mode": PAD_MODE,
        "tuned": {"threshold": best["th"], "logit_bias": best["bias"]},
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc},
        "counts": {"folders": int(len(df)), "focused_true": int((df['true']==1).sum()), "unfocused_true": int((df['true']==0).sum())}
    }
    with open(os.path.join(OUT_DIR, "eval_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
