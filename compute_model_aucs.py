import torch
ckpt = torch.load("best_model.pth", map_location="cpu")
print("checkpoint keys:", ckpt.keys())