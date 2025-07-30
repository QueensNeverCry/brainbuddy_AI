import pickle


with open("cnn_features/features/D_train.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data["features"][0]))  # ➤ torch.Tensor 여야 함
print(data["features"][0].shape)  # ➤ (T=100, D=1280) 정도 나와야 함