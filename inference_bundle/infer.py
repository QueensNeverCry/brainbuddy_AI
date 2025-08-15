import os
import glob
import argparse
import cv2
import torch
from model import CNN_LSTM
from preprocess import build_eval_tfms, to_tensor_from_bgr

def load_model(ckpt_path: str, device=None, backbone="resnet18", hidden=256, num_layers=2,
               bidirectional=True, dropout=0.3):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(backbone=backbone, hidden=hidden, num_layers=num_layers,
                     bidirectional=bidirectional, dropout=dropout).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

@torch.inference_mode()
def predict_sequence(model, device, frames_bgr, img_size=224, threshold=0.25, logit_bias=0.2, auto_zoom=True):
    tfms = build_eval_tfms(img_size)
    tens = [to_tensor_from_bgr(f, tfms, auto_zoom=auto_zoom) for f in frames_bgr]
    x = torch.stack(tens, dim=0).unsqueeze(0).to(device)  # (1,T,3,H,W)
    logit = model(x).float() + logit_bias
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)
    return pred, float(prob)

def load_folder_frames(folder: str, seq_len: int = 30):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    if len(files) < seq_len:
        if not files:
            raise RuntimeError(f"No images found in: {folder}")
        files = files + [files[-1]] * (seq_len - len(files))
    else:
        files = files[:seq_len]
    frames = [cv2.imread(fp) for fp in files]
    frames = [f for f in frames if f is not None]
    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from: {folder}")
    if len(frames) < seq_len:
        frames += [frames[-1]] * (seq_len - len(frames))
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to the_best.pth")
    ap.add_argument("--folder", required=True, help="Folder with ~30 frames")
    ap.add_argument("--seq-len", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--logit-bias", type=float, default=0.2)
    args = ap.parse_args()

    model, device = load_model(args.ckpt)
    frames = load_folder_frames(args.folder, seq_len=args.seq_len)
    pred, prob = predict_sequence(model, device, frames, img_size=args.img_size,
                                  threshold=args.threshold, logit_bias=args.logit_bias, auto_zoom=True)
    print(f"Prediction: {pred} (prob={prob:.4f})")

if __name__ == "__main__":
    main()

