import torch
import numpy as np

print("Numpy version:", np.__version__)
x = np.array([1, 2, 3])
print("From numpy:", torch.from_numpy(x))  # 오류 없으면 성공
