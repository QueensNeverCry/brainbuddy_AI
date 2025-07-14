import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.concent_model import EngagementModel
from videoframe_dataset import VideoEngagementDataset
import pickle
from tqdm import tqdm

def collate_fn(batch):
    # batch 내의 None 제거
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

def train():
    # [(영상경로, 라벨)] 리스트 가져오기
    with open("./preprocessed/dataset_link.pkl", "rb") as f:
        train_link = pickle.load(f)
    with open("./preprocessed/valdataset_link.pkl", "rb") as f:
        val_link = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available. Using CPU.")

    train_dataset = VideoEngagementDataset(train_link, T=10, device=device)
    val_dataset   = VideoEngagementDataset(val_link, T=10, device=device)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=collate_fn)#num_workers =4로 했더니 pickle 에러 발생
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_fn)

    # model
    model = EngagementModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in loop:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

# ✅ Windows-safe 진입점
if __name__ == '__main__':
    train()
