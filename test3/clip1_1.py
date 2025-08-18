# clip_zeroshot_labeler.py
# ------------------------------------------------------------
# CLIP 제로샷으로 하위 폴더(=30프레임 시퀀스 가정)를 라벨링해
# labels_zeroshot.csv를 생성합니다.
# - 네가 준 ROOT_DIRS를 그대로 사용
# - 폴더마다 이미지를 균등 샘플링(최대 MAX_FRAMES) 후 임베딩 평균
# - focused 확률(p_focused), margin, predicted_label(1/0) 기록
# ------------------------------------------------------------

import os, math, time
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip

# ====== 경로 설정: 네가 준 폴더들 (하위 폴더 전부 스캔) ======
ROOT_DIRS = [
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop",
]

# ====== 하이퍼파라미터 / 옵션 ======
MODEL_NAME = "ViT-B-32"       # open_clip 모델명 (예: ViT-B-32, ViT-L-14)
PRETRAINED = "openai"         # 체크포인트 (예: openai, laion2b_s34b_b79k)
MIN_FRAMES = 8                # 시퀀스로 인정할 최소 이미지 수
MAX_FRAMES = 30               # 폴더에서 균등 샘플 수(0=제한 없음, 권장:30)
BATCH_SIZE = 32               # 인코딩 배치 크기
THRESHOLD = 0.5               # focused 판정 임계값 (p_focused >= THRESHOLD → 1)
OUT_CSV = "labels_zeroshot.csv"

# 분류 대상 프롬프트들(원하면 수정 가능; 한/영 혼용 OK)
POS_PROMPTS = [
    "이 사람은 화면에 집중하고 있다.",
    "The person is focusing on the screen.",
    "시선이 목표를 향하고 있다.",
    "고개가 정면이며 주의가 흐트러지지 않았다.",
]
NEG_PROMPTS = [
    "이 사람은 산만하거나 딴짓을 하고 있다.",
    "The person is distracted or unfocused.",
    "시선이 다른 곳을 본다.",
    "고개가 옆으로 돌아가 있거나 멍해 보인다.",
]

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def list_sequence_folders(roots: List[str]) -> List[str]:
    """ROOT_DIRS 하위에서 이미지를 포함한 모든 폴더를 수집."""
    seqs = []
    for rd in roots:
        rd = os.path.abspath(rd)
        if not os.path.exists(rd):
            continue
        for cur, dirs, files in os.walk(rd):
            if any(f.lower().endswith(IMG_EXTS) for f in files):
                seqs.append(cur)
    # 중복 제거 + 정렬
    return sorted(set(seqs))

def list_images(folder: str) -> List[str]:
    try:
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(IMG_EXTS)
        )
        return [os.path.join(folder, f) for f in files]
    except Exception:
        return []

def even_sample(items: List[str], k: int) -> List[str]:
    """총 N개에서 k개를 균등 간격으로 샘플링 (앞쪽 치우침 방지)."""
    if k <= 0 or len(items) <= k:
        return items
    idxs = np.linspace(0, len(items)-1, k, dtype=int)
    return [items[i] for i in idxs]

def load_model(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess

@torch.no_grad()
def encode_texts(model, tokenizer, prompts: List[str], device: str) -> torch.Tensor:
    toks = tokenizer(prompts)
    t = model.encode_text(toks.to(device))
    return t / t.norm(dim=-1, keepdim=True)

@torch.no_grad()
def encode_images(model, preprocess, paths: List[str], device: str, batch_size: int = 32) -> torch.Tensor:
    xs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            xs.append(preprocess(img))
        except Exception:
            # 손상 파일은 스킵
            pass
    if not xs:
        return torch.empty(0, model.visual.output_dim if hasattr(model, "visual") else 512)
    X = torch.stack(xs, 0).to(device)

    embs = []
    for i in range(0, len(X), batch_size):
        chunk = X[i:i+batch_size]
        v = model.encode_image(chunk)
        v = v / v.norm(dim=-1, keepdim=True)
        embs.append(v)
    return torch.cat(embs, 0)  # (n_valid, d)

def prob_and_margin(image_embs: torch.Tensor, pos_txt: torch.Tensor, neg_txt: torch.Tensor):
    if image_embs.numel() == 0:
        return None, None
    pos_sim = (image_embs @ pos_txt.T).mean().item()
    neg_sim = (image_embs @ neg_txt.T).mean().item()
    # softmax 느낌의 확률화
    exp_pos, exp_neg = math.exp(pos_sim), math.exp(neg_sim)
    p_focused = exp_pos / (exp_pos + exp_neg + 1e-8)
    margin = pos_sim - neg_sim
    return float(p_focused), float(margin)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    model, tokenizer, preprocess = load_model(MODEL_NAME, PRETRAINED, device)
    pos_txt = encode_texts(model, tokenizer, POS_PROMPTS, device)
    neg_txt = encode_texts(model, tokenizer, NEG_PROMPTS, device)

    seq_folders = list_sequence_folders(ROOT_DIRS)
    if not seq_folders:
        print("[!] 유효한 이미지 폴더를 찾지 못했습니다.")
        return

    rows = []
    t0 = time.time()
    for folder in tqdm(seq_folders, desc="Zero-shot labeling"):
        imgs = list_images(folder)
        if len(imgs) < MIN_FRAMES:
            continue
        sel = even_sample(imgs, MAX_FRAMES) if MAX_FRAMES > 0 else imgs
        embs = encode_images(model, preprocess, sel, device, BATCH_SIZE)
        if embs.numel() == 0:
            continue
        p, m = prob_and_margin(embs, pos_txt, neg_txt)
        if p is None:
            continue
        pred = 1 if p >= THRESHOLD else 0  # 1=focused, 0=unfocused

        rows.append({
            "unit": "folder",
            "path": folder,                  # 대표 폴더 경로
            "folder": folder,
            "n_frames_agg": int(embs.shape[0]),
            "p_focused": p,
            "margin": m,
            "predicted_label": int(pred),
            "engine": f"{MODEL_NAME}/{PRETRAINED}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    if not rows:
        print("[!] 결과가 비었습니다. MIN_FRAMES/확장자/경로를 확인하세요.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {OUT_CSV} (rows={len(df)})")
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
